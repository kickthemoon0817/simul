"""
Isaac Sim MCP tools for Simul MCP Server.

This module provides granular Isaac Sim tools over the repo-owned typed bridge
when available, falling back to raw script execution only when required by the
current coverage. For legacy compatibility, raw script execution can still use
the stock VS Code socket when bridge usage is disabled.
"""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ...adapters import IsaacSocketClient, ScriptResult
from ...config import Settings, get_settings
from ...logging import LoggerMixin, get_logger
from ..schemas import ErrorResponse

logger = get_logger(__name__)


class IsaacTools(LoggerMixin):
    """
    Tools for Isaac Sim runtime operations via TCP socket.

    Each public method prefers a typed bridge action when one exists. If the
    action is unavailable, the method can fall back to raw script execution
    according to its routing policy.
    """

    def __init__(
        self,
        client: IsaacSocketClient,
        settings: Optional[Settings] = None,
        client_resolver: Optional[Callable[[], IsaacSocketClient]] = None,
    ) -> None:
        """
        Initialize Isaac Sim tools.

        Args:
            client: Pre-configured IsaacSocketClient for TCP communication.
            settings: Configuration settings.
            client_resolver: Optional callable used to resolve the target client
                dynamically per request/session.
        """
        self._default_client = client
        self._client_resolver = client_resolver
        self.settings = settings or get_settings()

    @property
    def _client(self) -> IsaacSocketClient:
        """Resolve the target client for the current request."""
        if self._client_resolver is not None:
            return self._client_resolver()
        return self._default_client

    @_client.setter
    def _client(self, client: IsaacSocketClient) -> None:
        """Update the default client when using non-session-scoped routing."""
        self._default_client = client

    async def _execute_json_script(
        self, script: str, transport_mode: str = "default"
    ) -> Dict[str, Any]:
        """
        Execute a Python script in Isaac Sim and parse JSON from stdout.

        The script MUST print exactly one JSON object via json.dumps().
        This helper handles connectivity errors, timeouts, parse failures,
        and script-level exceptions uniformly.

        Args:
            script: Python source code that prints a JSON object to stdout.
            transport_mode: Transport routing -- "default" uses client's preferred
                transport, "bridge_only" forces the bridge, "vscode_only" forces
                the VS Code socket.

        Returns:
            Parsed dict from stdout JSON, or an ErrorResponse dict.
        """
        try:
            if transport_mode == "bridge_only":
                result = await self._client.execute_bridge_script_only(script)
            elif transport_mode == "vscode_only":
                result = await self._client.execute_vscode_only(script)
            else:
                result = await self._client.execute(script)
        except ConnectionRefusedError as exc:
            return ErrorResponse(
                error=str(exc),
                error_type="ConnectionError",
            ).model_dump()
        except TimeoutError as exc:
            return ErrorResponse(
                error=str(exc),
                error_type="TimeoutError",
            ).model_dump()
        except Exception as exc:
            logger.error("Script execution failed: %s", exc, exc_info=True)
            return ErrorResponse(
                error=str(exc), error_type="Exception"
            ).model_dump()

        if not result.success:
            return ErrorResponse(
                error=result.error_value or "Script execution failed",
                error_type=result.error_name or "RuntimeError",
                details=(
                    {"traceback": result.traceback}
                    if result.traceback
                    else None
                ),
            ).model_dump()

        output = result.output.strip()
        if not output:
            return ErrorResponse(
                error="Script produced no output",
                error_type="EmptyOutput",
            ).model_dump()

        try:
            data: Dict[str, Any] = json.loads(output)
            data["success"] = True
            return data
        except json.JSONDecodeError as exc:
            return ErrorResponse(
                error=f"Failed to parse script output as JSON: {exc}",
                error_type="JSONDecodeError",
                details={"raw_output": output[:2000]},
            ).model_dump()

    async def _execute_bridge_action(
        self, action: str, payload: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a typed bridge action without falling back to VS Code.

        Returns:
            Parsed payload dict on success, an error dict on bridge-only
            failure, or None when the action is unavailable and the caller
            should try a bridge-script fallback.
        """
        if not self._client.bridge_enabled:
            return None

        try:
            response = await self._client.bridge_request(action, payload or {})
        except (ConnectionRefusedError, TimeoutError, OSError, ValueError) as exc:
            return ErrorResponse(
                error=str(exc),
                error_type=type(exc).__name__,
            ).model_dump()

        if response.get("status") == "ok":
            result_payload = response.get("payload", {})
            if not isinstance(result_payload, dict):
                return ErrorResponse(
                    error="Bridge response payload must be an object.",
                    error_type="BridgeProtocolError",
                ).model_dump()
            result_payload.setdefault("success", True)
            return result_payload

        error = response.get("error", {})
        error_name = str(error.get("name", "BridgeError"))
        if error_name == "UnknownAction":
            return None
        return ErrorResponse(
            error=str(error.get("message", "Bridge request failed")),
            error_type=error_name,
            details={"traceback": error.get("traceback")} if error.get("traceback") else None,
        ).model_dump()

    @property
    def _raw_script_transport_mode(self) -> str:
        """Route raw-script fallbacks away from the typed bridge by default."""
        if self._client.bridge_enabled:
            return "vscode_only"
        return "default"

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
        max_depth: int = -1,
        max_items: int = 500,
    ) -> Dict[str, Any]:
        """
        List prims in the current Isaac Sim stage with optional filtering.

        Args:
            root_path: Root prim path to start traversal from.
            prim_type: Filter by USD prim type name (e.g. "Mesh", "Xform").
            max_depth: Maximum traversal depth (-1 for unlimited).
            max_items: Maximum number of prims to return (1–10000).

        Returns:
            Dict with list of prim entries (path, type, name, active).
        """
        max_items = max(1, min(max_items, 10000))
        bridge_result = await self._execute_bridge_action(
            "list_prims",
            {
                "root_path": root_path,
                "prim_type": prim_type,
                "max_depth": max_depth,
                "max_items": max_items,
            },
        )
        if bridge_result is not None:
            return bridge_result

        _root_path = json.dumps(root_path)
        _prim_type = json.dumps(prim_type or "")
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
                    max_n = {max_items}
                    root_depth = len(str(root.GetPath()).rstrip("/").split("/"))

                    for p in Usd.PrimRange(root):
                        path_str = str(p.GetPath())
                        depth = len(path_str.rstrip("/").split("/")) - root_depth
                        if max_d >= 0 and depth > max_d:
                            continue
                        ptype = p.GetTypeName()
                        if type_filter and ptype != type_filter:
                            continue
                        prims.append({{
                            "path": path_str,
                            "type": ptype,
                            "name": p.GetName(),
                            "active": p.IsActive(),
                        }})
                        if len(prims) >= max_n:
                            break
                    print(json.dumps({{
                        "root_path": {_root_path},
                        "type_filter": type_filter or None,
                        "count": len(prims),
                        "truncated": len(prims) >= max_n,
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

        _prim_path = json.dumps(prim_path)
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

                    attrs = {{}}
                    for attr in prim.GetAttributes():
                        try:
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

        _prim_path = json.dumps(prim_path)
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
        search_type: str = "type",
        query: str = "Mesh",
        root_path: str = "/",
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """
        Search for prims by type name or name pattern.

        Args:
            search_type: "type" to match prim type, "name" to match prim name.
            query: Type name (e.g. "Mesh") or name substring to search for.
            root_path: Root path to search under.
            max_results: Maximum results to return.

        Returns:
            Dict with matching prim paths and types.
        """
        max_results = max(1, min(max_results, 10000))
        bridge_result = await self._execute_bridge_action(
            "search_prims",
            {
                "search_type": search_type,
                "query": query,
                "root_path": root_path,
                "max_results": max_results,
            },
        )
        if bridge_result is not None:
            return bridge_result

        _root_path = json.dumps(root_path)
        _search_type = json.dumps(search_type)
        _query = json.dumps(query)
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
                    for p in Usd.PrimRange(root):
                        if search_type == "type":
                            if p.GetTypeName() == query:
                                matches.append({{"path": str(p.GetPath()), "type": p.GetTypeName(), "name": p.GetName()}})
                        elif search_type == "name":
                            if query.lower() in p.GetName().lower():
                                matches.append({{"path": str(p.GetPath()), "type": p.GetTypeName(), "name": p.GetName()}})
                        if len(matches) >= max_r:
                            break
                    print(json.dumps({{
                        "search_type": search_type,
                        "query": query,
                        "root_path": {_root_path},
                        "count": len(matches),
                        "truncated": len(matches) >= max_r,
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
        """
        script = textwrap.dedent("""\
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

                for p in stage.Traverse():
                    total += 1
                    t = p.GetTypeName() or "Typeless"
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

    # ------------------------------------------------------------------
    # Phase 2: Viewport & Camera
    # ------------------------------------------------------------------

    async def get_isaac_camera_info(
        self, camera_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get active or specified camera parameters.

        Args:
            camera_path: USD path to a camera prim. None uses active viewport camera.

        Returns:
            Dict with camera position, target, focal length, clipping range, etc.
        """
        _cam_path = json.dumps(camera_path or "")
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                cam_path = {_cam_path}
                cam_prim = None
                if cam_path:
                    cam_prim = stage.GetPrimAtPath(cam_path)
                    if not cam_prim.IsValid() or not cam_prim.IsA(UsdGeom.Camera):
                        print(json.dumps({{"error": f"Not a valid Camera prim: {{cam_path}}"}}))
                        cam_prim = None
                else:
                    try:
                        import omni.kit.viewport.utility as vp_util
                        vp_api = vp_util.get_active_viewport()
                        if vp_api:
                            cam_path = str(vp_api.camera_path)
                            cam_prim = stage.GetPrimAtPath(cam_path)
                    except Exception:
                        for p in stage.Traverse():
                            if p.IsA(UsdGeom.Camera):
                                cam_prim = p
                                cam_path = str(p.GetPath())
                                break

                if cam_prim and cam_prim.IsValid():
                    cam = UsdGeom.Camera(cam_prim)
                    gf_cam = cam.GetCamera(Usd.TimeCode.Default())
                    xformable = UsdGeom.Xformable(cam_prim)
                    world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    pos = world_xform.ExtractTranslation()
                    rot = world_xform.ExtractRotation().GetQuat()
                    print(json.dumps({{
                        "camera_path": str(cam_path),
                        "position": list(pos),
                        "rotation_quat": [float(rot.GetReal())] + [float(x) for x in rot.GetImaginary()],
                        "focal_length": float(cam.GetFocalLengthAttr().Get()),
                        "horizontal_aperture": float(cam.GetHorizontalApertureAttr().Get()),
                        "vertical_aperture": float(cam.GetVerticalApertureAttr().Get()),
                        "clipping_range": [float(x) for x in cam.GetClippingRangeAttr().Get()],
                        "projection": str(cam.GetProjectionAttr().Get()),
                    }}))
                elif not cam_path:
                    print(json.dumps({{"error": "No camera found in scene"}}))
        """)
        return await self._execute_json_script(script)

    async def list_isaac_cameras(self) -> Dict[str, Any]:
        """
        List all camera prims in the current Isaac Sim stage.

        Returns:
            Dict with list of cameras with paths and basic parameters.
        """
        script = textwrap.dedent("""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            else:
                cameras = []
                for p in stage.Traverse():
                    if p.IsA(UsdGeom.Camera):
                        cam = UsdGeom.Camera(p)
                        cameras.append({
                            "path": str(p.GetPath()),
                            "name": p.GetName(),
                            "focal_length": cam.GetFocalLengthAttr().Get(),
                            "projection": cam.GetProjectionAttr().Get(),
                        })
                active_cam = None
                try:
                    import omni.kit.viewport.utility as vp_util
                    vp_api = vp_util.get_active_viewport()
                    if vp_api:
                        active_cam = str(vp_api.camera_path)
                except Exception:
                    pass
                print(json.dumps({
                    "count": len(cameras),
                    "active_camera": active_cam,
                    "cameras": cameras,
                }))
        """)
        return await self._execute_json_script(script)

    async def set_isaac_camera(
        self,
        position: Optional[List[float]] = None,
        target: Optional[List[float]] = None,
        camera_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Set camera position and/or look-at target.

        Args:
            position: Camera position as [x, y, z].
            target: Look-at target position as [x, y, z].
            camera_path: Path to camera prim. None uses active viewport camera.

        Returns:
            Dict confirming the updated camera state.
        """
        pos_str = str(position) if position else "None"
        tgt_str = str(target) if target else "None"
        _cam_path = json.dumps(camera_path or "")
        script = textwrap.dedent(f"""\
            import json
            from pxr import Gf, Usd, UsdGeom

            pos = {pos_str}
            tgt = {tgt_str}

            try:
                import omni.kit.viewport.utility as vp_util
                from omni.kit.viewport.utility.camera_state import ViewportCameraState

                cam_path = {_cam_path}
                vp_api = vp_util.get_active_viewport()
                if vp_api is None:
                    print(json.dumps({{"error": "No active viewport"}}))
                else:
                    if cam_path:
                        vp_api.camera_path = cam_path
                    cam_path = str(vp_api.camera_path)

                    state = ViewportCameraState(viewport=vp_api)
                    if pos is not None:
                        state.set_position_world(Gf.Vec3d(*pos), True)
                    if tgt is not None:
                        state.set_target_world(Gf.Vec3d(*tgt), True)

                    # Read back the new state
                    new_pos = state.position_world
                    new_tgt = state.target_world
                    print(json.dumps({{
                        "camera_path": cam_path,
                        "position": [float(x) for x in new_pos],
                        "target": [float(x) for x in new_tgt],
                    }}))
            except ImportError:
                # Fallback: direct USD edit for headless mode
                import omni.usd
                stage = omni.usd.get_context().get_stage()
                if stage is None:
                    print(json.dumps({{"error": "No stage is currently open"}}))
                else:
                    cam_path = {_cam_path}
                    if not cam_path:
                        for p in stage.Traverse():
                            if p.IsA(UsdGeom.Camera):
                                cam_path = str(p.GetPath())
                                break
                    if not cam_path:
                        print(json.dumps({{"error": "No camera found"}}))
                    else:
                        prim = stage.GetPrimAtPath(cam_path)
                        if not prim.IsValid():
                            print(json.dumps({{"error": f"Camera prim not found: {{cam_path}}"}}))
                        else:
                            xformable = UsdGeom.Xformable(prim)
                            ops = xformable.GetOrderedXformOps()
                            if pos is not None:
                                for op in ops:
                                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                                        op.Set(Gf.Vec3d(*pos))
                                        break
                            xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                            new_pos = xform.ExtractTranslation()
                            print(json.dumps({{
                                "camera_path": cam_path,
                                "position": [float(x) for x in new_pos],
                            }}))
        """)
        return await self._execute_json_script(script)

    async def capture_isaac_viewport(
        self,
        width: int = 1280,
        height: int = 720,
    ) -> Dict[str, Any]:
        """
        Capture the active viewport as a base64-encoded PNG image.

        Args:
            width: Output image width in pixels (1–7680).
            height: Output image height in pixels (1–7680).

        Returns:
            Dict with base64 image data and dimensions.
        """
        width = max(1, min(width, 7680))
        height = max(1, min(height, 7680))
        script = textwrap.dedent(f"""\
            import json
            import base64
            import os
            import tempfile
            try:
                import omni.kit.viewport.utility as vp_util
                import omni.kit.app

                vp_api = vp_util.get_active_viewport()
                if vp_api is None:
                    print(json.dumps({{"error": "No active viewport found"}}))
                else:
                    tmp_path = os.path.join(tempfile.gettempdir(), "_simul_mcp_capture.png")
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                    # Set requested resolution on the viewport
                    vp_api.resolution = ({width}, {height})
                    # Let the viewport re-render at the new resolution
                    for _ in range(4):
                        await omni.kit.app.get_app().next_update_async()

                    # Kit 106+ (Isaac Sim 5.x): use schedule_capture with FileCapture
                    from omni.kit.widget.viewport.capture import FileCapture
                    capture = FileCapture(tmp_path)
                    vp_api.schedule_capture(capture)

                    # Wait for the file to be written
                    for _ in range(60):
                        await omni.kit.app.get_app().next_update_async()
                        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                            break

                    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                        with open(tmp_path, "rb") as f:
                            data = base64.b64encode(f.read()).decode("ascii")
                        os.remove(tmp_path)
                        print(json.dumps({{
                            "width": {width},
                            "height": {height},
                            "format": "png",
                            "encoding": "base64",
                            "image_base64": data,
                        }}))
                    else:
                        print(json.dumps({{"error": "Viewport capture failed — file not created"}}))
            except ImportError as e:
                print(json.dumps({{"error": f"Viewport capture not available: {{e}}"}}))
            except Exception as e:
                print(json.dumps({{"error": f"Viewport capture error: {{e}}"}}))
        """)
        return await self._execute_json_script(script)


    # ------------------------------------------------------------------
    # Phase 3: Prim Manipulation
    # ------------------------------------------------------------------

    async def create_isaac_prim(
        self,
        prim_path: str,
        prim_type: str = "Xform",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new prim in the current Isaac Sim stage.

        Args:
            prim_path: USD path for the new prim (e.g. "/World/MyObject").
            prim_type: USD type name (e.g. "Xform", "Mesh", "Cube", "Sphere").
            attributes: Optional dict of attribute name→value to set on creation.

        Returns:
            Dict confirming the created prim path and type.
        """
        _prim_path = json.dumps(prim_path)
        _prim_type = json.dumps(prim_type)
        _attrs_str = json.dumps(json.dumps(attributes)) if attributes else '"{}"'
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                existing = stage.GetPrimAtPath({_prim_path})
                if existing.IsValid():
                    print(json.dumps({{"error": "Prim already exists: " + {_prim_path}}}))
                else:
                    prim = stage.DefinePrim({_prim_path}, {_prim_type})
                    if not prim.IsValid():
                        print(json.dumps({{"error": "Failed to create prim: " + {_prim_path}}}))
                    else:
                        attrs = json.loads({_attrs_str})
                        for name, val in attrs.items():
                            attr = prim.GetAttribute(name)
                            if attr.IsValid():
                                attr.Set(val)
                        print(json.dumps({{
                            "prim_path": str(prim.GetPath()),
                            "prim_type": prim.GetTypeName(),
                            "created": True,
                        }}))
        """)
        return await self._execute_json_script(script)

    async def delete_isaac_prim(self, prim_path: str) -> Dict[str, Any]:
        """
        Delete a prim and its children from the current stage.

        Args:
            prim_path: USD path of the prim to delete.

        Returns:
            Dict confirming deletion.
        """
        _prim_path = json.dumps(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                else:
                    edit = Sdf.BatchNamespaceEdit()
                    edit.Add(Sdf.NamespaceEdit.Remove({_prim_path}))
                    if stage.GetRootLayer().Apply(edit):
                        print(json.dumps({{"prim_path": {_prim_path}, "deleted": True}}))
                    else:
                        print(json.dumps({{"error": "Failed to delete prim: " + {_prim_path}}}))
        """)
        return await self._execute_json_script(script)

    async def set_isaac_prim_transform(
        self,
        prim_path: str,
        translation: Optional[List[float]] = None,
        rotation_euler: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Set the transform of a prim (translation, rotation, scale).

        Args:
            prim_path: USD path of the prim.
            translation: Position as [x, y, z].
            rotation_euler: Rotation in degrees as [x, y, z] (XYZ Euler).
            scale: Scale as [x, y, z].

        Returns:
            Dict with updated transform values.
        """
        _prim_path = json.dumps(prim_path)
        t_str = str(translation) if translation else "None"
        r_str = str(rotation_euler) if rotation_euler else "None"
        s_str = str(scale) if scale else "None"
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
                    t = {t_str}
                    r = {r_str}
                    s = {s_str}
                    if t is not None:
                        found = False
                        for op in xformable.GetOrderedXformOps():
                            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                                op.Set(Gf.Vec3d(*t))
                                found = True
                                break
                        if not found:
                            xformable.AddTranslateOp().Set(Gf.Vec3d(*t))
                    if r is not None:
                        found = False
                        for op in xformable.GetOrderedXformOps():
                            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                                op.Set(Gf.Vec3f(*r))
                                found = True
                                break
                        if not found:
                            xformable.AddRotateXYZOp().Set(Gf.Vec3f(*r))
                    if s is not None:
                        found = False
                        for op in xformable.GetOrderedXformOps():
                            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                                op.Set(Gf.Vec3f(*s))
                                found = True
                                break
                        if not found:
                            xformable.AddScaleOp().Set(Gf.Vec3f(*s))
                    # Read back
                    xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    pos = xform.ExtractTranslation()
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "translation": list(pos),
                        "rotation_euler_set": r,
                        "scale_set": s,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def set_isaac_prim_visibility(
        self, prim_path: str, visible: bool
    ) -> Dict[str, Any]:
        """
        Set the visibility of a prim.

        Args:
            prim_path: USD path of the prim.
            visible: True to make visible, False to hide.

        Returns:
            Dict confirming the visibility state.
        """
        _prim_path = json.dumps(prim_path)
        _vis_token = json.dumps("inherited" if visible else "invisible")
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
                elif not prim.IsA(UsdGeom.Imageable):
                    print(json.dumps({{"error": "Prim is not Imageable: " + {_prim_path}}}))
                else:
                    img = UsdGeom.Imageable(prim)
                    img.GetVisibilityAttr().Set({_vis_token})
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "visibility": {_vis_token},
                        "effective_visibility": img.ComputeVisibility(),
                    }}))
        """)
        return await self._execute_json_script(script)

    async def set_isaac_prim_attribute(
        self,
        prim_path: str,
        attribute_name: str,
        value: Any,
    ) -> Dict[str, Any]:
        """
        Set a single attribute value on a prim.

        Args:
            prim_path: USD path of the prim.
            attribute_name: Name of the attribute to set.
            value: Value to set. Must be JSON-serializable.

        Returns:
            Dict confirming the attribute was set.
        """
        _prim_path = json.dumps(prim_path)
        _attr_name = json.dumps(attribute_name)
        _val_str = json.dumps(json.dumps(value))
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
                    attr = prim.GetAttribute({_attr_name})
                    if not attr.IsValid():
                        print(json.dumps({{"error": "Attribute not found: " + {_attr_name} + " on " + {_prim_path}}}))
                    else:
                        val = json.loads({_val_str})
                        try:
                            attr.Set(val)
                            read_back = attr.Get()
                            print(json.dumps({{
                                "prim_path": {_prim_path},
                                "attribute": {_attr_name},
                                "value_set": val,
                                "value_read": str(read_back),
                            }}))
                        except Exception as e:
                            print(json.dumps({{"error": f"Failed to set attribute: {{e}}"}}))
        """)
        return await self._execute_json_script(script)

    async def duplicate_isaac_prim(
        self, prim_path: str, new_path: str
    ) -> Dict[str, Any]:
        """
        Duplicate a prim to a new path.

        Args:
            prim_path: Source prim path.
            new_path: Destination prim path.

        Returns:
            Dict confirming the duplication.
        """
        _prim_path = json.dumps(prim_path)
        _new_path = json.dumps(new_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                src = stage.GetPrimAtPath({_prim_path})
                if not src.IsValid():
                    print(json.dumps({{"error": "Source prim not found: " + {_prim_path}}}))
                else:
                    dst = stage.GetPrimAtPath({_new_path})
                    if dst.IsValid():
                        print(json.dumps({{"error": "Destination already exists: " + {_new_path}}}))
                    else:
                        Sdf.CopySpec(
                            stage.GetRootLayer(),
                            Sdf.Path({_prim_path}),
                            stage.GetRootLayer(),
                            Sdf.Path({_new_path}),
                        )
                        new_prim = stage.GetPrimAtPath({_new_path})
                        if new_prim.IsValid():
                            print(json.dumps({{
                                "source_path": {_prim_path},
                                "new_path": {_new_path},
                                "type": new_prim.GetTypeName(),
                                "duplicated": True,
                            }}))
                        else:
                            print(json.dumps({{"error": "Duplication failed"}}))
        """)
        return await self._execute_json_script(script)

    async def reparent_isaac_prim(
        self, prim_path: str, new_parent_path: str
    ) -> Dict[str, Any]:
        """
        Move a prim under a new parent.

        Args:
            prim_path: USD path of the prim to move.
            new_parent_path: USD path of the new parent prim.

        Returns:
            Dict with the new full path of the reparented prim.
        """
        _prim_path = json.dumps(prim_path)
        _new_parent = json.dumps(new_parent_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                else:
                    parent = stage.GetPrimAtPath({_new_parent})
                    if not parent.IsValid():
                        print(json.dumps({{"error": "Parent not found: " + {_new_parent}}}))
                    else:
                        name = prim.GetName()
                        new_full_path = {_new_parent} + "/" + name
                        edit = Sdf.BatchNamespaceEdit()
                        edit.Add(
                            Sdf.NamespaceEdit.Reparent(
                                Sdf.Path({_prim_path}),
                                Sdf.Path({_new_parent}),
                                -1,
                            )
                        )
                        if stage.GetRootLayer().Apply(edit):
                            print(json.dumps({{
                                "old_path": {_prim_path},
                                "new_path": new_full_path,
                                "parent": {_new_parent},
                                "reparented": True,
                            }}))
                        else:
                            print(json.dumps({{"error": "Reparent operation failed"}}))
        """)
        return await self._execute_json_script(script)

    # ------------------------------------------------------------------
    # Phase 4: Physics Inspection
    # ------------------------------------------------------------------

    async def get_isaac_physics_scene(self) -> Dict[str, Any]:
        """
        Get physics scene configuration from the current stage.

        Returns:
            Dict with gravity, solver settings, and physics scene prim paths.
        """
        script = textwrap.dedent("""\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            else:
                scenes = []
                for p in stage.Traverse():
                    if p.IsA(UsdPhysics.Scene):
                        scene = UsdPhysics.Scene(p)
                        grav = scene.GetGravityDirectionAttr().Get()
                        mag = scene.GetGravityMagnitudeAttr().Get()
                        scenes.append({
                            "path": str(p.GetPath()),
                            "gravity_direction": list(grav) if grav else None,
                            "gravity_magnitude": mag,
                        })
                print(json.dumps({
                    "physics_scene_count": len(scenes),
                    "scenes": scenes,
                    "has_physics": len(scenes) > 0,
                }))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_rigid_body_info(
        self, prim_path: str
    ) -> Dict[str, Any]:
        """
        Get rigid body physics properties of a prim.

        Args:
            prim_path: USD path of the prim with RigidBodyAPI applied.

        Returns:
            Dict with mass, velocity, angular velocity, kinematic state, etc.
        """
        _prim_path = json.dumps(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                elif not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    print(json.dumps({{"error": "Prim does not have RigidBodyAPI: " + {_prim_path}}}))
                else:
                    rb = UsdPhysics.RigidBodyAPI(prim)
                    vel = rb.GetVelocityAttr().Get()
                    ang_vel = rb.GetAngularVelocityAttr().Get()
                    kinematic = rb.GetKinematicEnabledAttr().Get()
                    rb_enabled = rb.GetRigidBodyEnabledAttr().Get()

                    mass_api = None
                    mass = None
                    com = None
                    inertia = None
                    if prim.HasAPI(UsdPhysics.MassAPI):
                        mass_api = UsdPhysics.MassAPI(prim)
                        mass = mass_api.GetMassAttr().Get()
                        com_attr = mass_api.GetCenterOfMassAttr().Get()
                        com = list(com_attr) if com_attr else None
                        inertia_attr = mass_api.GetDiagonalInertiaAttr().Get()
                        inertia = list(inertia_attr) if inertia_attr else None

                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "rigid_body_enabled": rb_enabled if rb_enabled is not None else True,
                        "is_kinematic": kinematic if kinematic is not None else False,
                        "velocity": list(vel) if vel else [0, 0, 0],
                        "angular_velocity": list(ang_vel) if ang_vel else [0, 0, 0],
                        "has_mass_api": prim.HasAPI(UsdPhysics.MassAPI),
                        "mass": mass,
                        "center_of_mass": com,
                        "diagonal_inertia": inertia,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def list_isaac_physics_objects(
        self, root_path: str = "/"
    ) -> Dict[str, Any]:
        """
        List all prims with physics APIs applied in the stage.

        Args:
            root_path: Root path to search under.

        Returns:
            Dict with lists of rigid bodies, colliders, and joints.
        """
        _root_path = json.dumps(root_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                root = stage.GetPrimAtPath({_root_path})
                rigid_bodies = []
                colliders = []
                joints = []
                for p in Usd.PrimRange(root):
                    path = str(p.GetPath())
                    if p.HasAPI(UsdPhysics.RigidBodyAPI):
                        rigid_bodies.append({{"path": path, "type": p.GetTypeName()}})
                    if p.HasAPI(UsdPhysics.CollisionAPI):
                        colliders.append({{"path": path, "type": p.GetTypeName()}})
                    if p.IsA(UsdPhysics.Joint):
                        joints.append({{"path": path, "type": p.GetTypeName()}})
                print(json.dumps({{
                    "root_path": {_root_path},
                    "rigid_body_count": len(rigid_bodies),
                    "collider_count": len(colliders),
                    "joint_count": len(joints),
                    "rigid_bodies": rigid_bodies[:200],
                    "colliders": colliders[:200],
                    "joints": joints[:200],
                }}))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_collision_info(
        self, prim_path: str
    ) -> Dict[str, Any]:
        """
        Get collision properties of a prim.

        Args:
            prim_path: USD path of the prim with CollisionAPI.

        Returns:
            Dict with collision enabled state and approximation type.
        """
        _prim_path = json.dumps(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                elif not prim.HasAPI(UsdPhysics.CollisionAPI):
                    print(json.dumps({{"error": "Prim does not have CollisionAPI: " + {_prim_path}}}))
                else:
                    col = UsdPhysics.CollisionAPI(prim)
                    enabled = col.GetCollisionEnabledAttr().Get()
                    # Check for mesh collision API
                    approx = None
                    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                        mesh_col = UsdPhysics.MeshCollisionAPI(prim)
                        approx = mesh_col.GetApproximationAttr().Get()
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "collision_enabled": enabled if enabled is not None else True,
                        "has_mesh_collision": prim.HasAPI(UsdPhysics.MeshCollisionAPI),
                        "approximation": approx,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_joint_info(
        self, prim_path: str
    ) -> Dict[str, Any]:
        """
        Get joint information for a physics joint prim.

        Args:
            prim_path: USD path of the joint prim.

        Returns:
            Dict with joint type, bodies, limits, and drive settings.
        """
        _prim_path = json.dumps(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                elif not prim.IsA(UsdPhysics.Joint):
                    print(json.dumps({{"error": "Prim is not a Joint: " + {_prim_path}}}))
                else:
                    joint = UsdPhysics.Joint(prim)
                    body0 = joint.GetBody0Rel().GetTargets()
                    body1 = joint.GetBody1Rel().GetTargets()
                    enabled = joint.GetJointEnabledAttr().Get()
                    exclude = joint.GetExcludeFromArticulationAttr().Get()
                    break_force = joint.GetBreakForceAttr().Get()
                    break_torque = joint.GetBreakTorqueAttr().Get()

                    # Check for revolute or prismatic limits
                    limits = {{}}
                    if prim.IsA(UsdPhysics.RevoluteJoint):
                        rev = UsdPhysics.RevoluteJoint(prim)
                        limits["type"] = "revolute"
                        limits["axis"] = rev.GetAxisAttr().Get()
                        limits["lower"] = rev.GetLowerLimitAttr().Get()
                        limits["upper"] = rev.GetUpperLimitAttr().Get()
                    elif prim.IsA(UsdPhysics.PrismaticJoint):
                        pri = UsdPhysics.PrismaticJoint(prim)
                        limits["type"] = "prismatic"
                        limits["axis"] = pri.GetAxisAttr().Get()
                        limits["lower"] = pri.GetLowerLimitAttr().Get()
                        limits["upper"] = pri.GetUpperLimitAttr().Get()

                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "joint_type": prim.GetTypeName(),
                        "enabled": enabled if enabled is not None else True,
                        "body0": [str(b) for b in body0] if body0 else [],
                        "body1": [str(b) for b in body1] if body1 else [],
                        "exclude_from_articulation": exclude,
                        "break_force": break_force,
                        "break_torque": break_torque,
                        "limits": limits if limits else None,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_mass_properties(
        self, prim_path: str
    ) -> Dict[str, Any]:
        """
        Get mass properties of a prim.

        Args:
            prim_path: USD path of the prim with MassAPI.

        Returns:
            Dict with mass, density, center of mass, and inertia tensor.
        """
        _prim_path = json.dumps(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                elif not prim.HasAPI(UsdPhysics.MassAPI):
                    print(json.dumps({{"error": "Prim does not have MassAPI: " + {_prim_path}}}))
                else:
                    mass_api = UsdPhysics.MassAPI(prim)
                    mass = mass_api.GetMassAttr().Get()
                    density = mass_api.GetDensityAttr().Get()
                    com = mass_api.GetCenterOfMassAttr().Get()
                    inertia = mass_api.GetDiagonalInertiaAttr().Get()
                    principal_axes = mass_api.GetPrincipalAxesAttr().Get()
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "mass": mass,
                        "density": density,
                        "center_of_mass": list(com) if com else None,
                        "diagonal_inertia": list(inertia) if inertia else None,
                        "principal_axes": [principal_axes.GetReal()] + list(principal_axes.GetImaginary()) if principal_axes else None,
                    }}))
        """)
        return await self._execute_json_script(script)

    # ------------------------------------------------------------------
    # Phase 5: Physics Configuration
    # ------------------------------------------------------------------

    async def add_isaac_rigid_body(
        self,
        prim_path: str,
        kinematic: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply RigidBodyAPI to a prim.

        Args:
            prim_path: USD path of the prim.
            kinematic: If True, makes the body kinematic (animated, not simulated).

        Returns:
            Dict confirming the API was applied.
        """
        _prim_path = json.dumps(prim_path)
        kin_str = "True" if kinematic else "False"
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                else:
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        print(json.dumps({{"error": "RigidBodyAPI already applied: " + {_prim_path}}}))
                    else:
                        UsdPhysics.RigidBodyAPI.Apply(prim)
                        if {kin_str}:
                            rb = UsdPhysics.RigidBodyAPI(prim)
                            rb.GetKinematicEnabledAttr().Set(True)
                        print(json.dumps({{
                            "prim_path": {_prim_path},
                            "rigid_body_applied": True,
                            "kinematic": {kin_str},
                        }}))
        """)
        return await self._execute_json_script(script)

    async def add_isaac_collision(
        self,
        prim_path: str,
        approximation: str = "none",
    ) -> Dict[str, Any]:
        """
        Apply CollisionAPI (and optionally MeshCollisionAPI) to a prim.

        Args:
            prim_path: USD path of the prim.
            approximation: Collision approximation type: "none", "convexHull",
                "convexDecomposition", "meshSimplification", "boundingSphere",
                "boundingCube".

        Returns:
            Dict confirming collision was added.
        """
        _prim_path = json.dumps(prim_path)
        _approx = json.dumps(approximation)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                else:
                    UsdPhysics.CollisionAPI.Apply(prim)
                    approx = {_approx}
                    if approx != "none":
                        UsdPhysics.MeshCollisionAPI.Apply(prim)
                        mesh_col = UsdPhysics.MeshCollisionAPI(prim)
                        mesh_col.GetApproximationAttr().Set(approx)
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "collision_applied": True,
                        "approximation": approx,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def set_isaac_mass_properties(
        self,
        prim_path: str,
        mass: Optional[float] = None,
        density: Optional[float] = None,
        center_of_mass: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Set mass properties on a prim (applies MassAPI if not present).

        Args:
            prim_path: USD path of the prim.
            mass: Mass value in kg.
            density: Density value.
            center_of_mass: Center of mass as [x, y, z].

        Returns:
            Dict confirming updated mass properties.
        """
        _prim_path = json.dumps(prim_path)
        m_str = str(mass) if mass is not None else "None"
        d_str = str(density) if density is not None else "None"
        com_str = str(center_of_mass) if center_of_mass else "None"
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdPhysics, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                else:
                    if not prim.HasAPI(UsdPhysics.MassAPI):
                        UsdPhysics.MassAPI.Apply(prim)
                    mass_api = UsdPhysics.MassAPI(prim)
                    m = {m_str}
                    d = {d_str}
                    com = {com_str}
                    if m is not None:
                        mass_api.GetMassAttr().Set(m)
                    if d is not None:
                        mass_api.GetDensityAttr().Set(d)
                    if com is not None:
                        mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*com))
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "mass": mass_api.GetMassAttr().Get(),
                        "density": mass_api.GetDensityAttr().Get(),
                        "center_of_mass": list(mass_api.GetCenterOfMassAttr().Get()) if mass_api.GetCenterOfMassAttr().Get() else None,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def set_isaac_physics_material(
        self,
        prim_path: str,
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
        restitution: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Create or update a physics material on a prim.

        Args:
            prim_path: USD path where the material should be created/updated.
            static_friction: Static friction coefficient.
            dynamic_friction: Dynamic friction coefficient.
            restitution: Restitution (bounciness) coefficient.

        Returns:
            Dict confirming the physics material was set.
        """
        _prim_path = json.dumps(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdShade, UsdPhysics, Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    # Create material prim
                    prim = stage.DefinePrim({_prim_path}, "Material")
                UsdPhysics.MaterialAPI.Apply(prim)
                mat = UsdPhysics.MaterialAPI(prim)
                mat.GetStaticFrictionAttr().Set({static_friction})
                mat.GetDynamicFrictionAttr().Set({dynamic_friction})
                mat.GetRestitutionAttr().Set({restitution})
                print(json.dumps({{
                    "prim_path": {_prim_path},
                    "static_friction": {static_friction},
                    "dynamic_friction": {dynamic_friction},
                    "restitution": {restitution},
                    "physics_material_applied": True,
                }}))
        """)
        return await self._execute_json_script(script)

    async def create_isaac_physics_scene(
        self,
        prim_path: str = "/World/PhysicsScene",
        gravity_direction: Optional[List[float]] = None,
        gravity_magnitude: float = 9.81,
    ) -> Dict[str, Any]:
        """
        Create a UsdPhysics.Scene prim with gravity settings.

        Args:
            prim_path: USD path for the physics scene prim.
            gravity_direction: Gravity direction vector, defaults to [0, 0, -1].
            gravity_magnitude: Gravity magnitude in m/s^2.

        Returns:
            Dict confirming the physics scene was created with its settings.
        """
        _prim_path = json.dumps(prim_path)
        grav_dir = gravity_direction or [0.0, 0.0, -1.0]
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdPhysics, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                existing = stage.GetPrimAtPath({_prim_path})
                if existing.IsValid() and existing.IsA(UsdPhysics.Scene):
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "already_existed": True,
                        "message": "Physics scene already exists at this path",
                    }}))
                else:
                    scene_prim = stage.DefinePrim({_prim_path}, "PhysicsScene")
                    if not scene_prim.IsValid():
                        print(json.dumps({{"error": "Failed to create physics scene prim"}}))
                    else:
                        scene = UsdPhysics.Scene(scene_prim)
                        grav_dir = Gf.Vec3f({grav_dir[0]}, {grav_dir[1]}, {grav_dir[2]})
                        scene.CreateGravityDirectionAttr(grav_dir)
                        scene.CreateGravityMagnitudeAttr({gravity_magnitude})
                        print(json.dumps({{
                            "prim_path": {_prim_path},
                            "gravity_direction": [{grav_dir[0]}, {grav_dir[1]}, {grav_dir[2]}],
                            "gravity_magnitude": {gravity_magnitude},
                            "created": True,
                        }}))
        """)
        return await self._execute_json_script(script)

    # ------------------------------------------------------------------
    # Phase 6: Simulation Control
    # ------------------------------------------------------------------

    async def get_isaac_simulation_state(self) -> Dict[str, Any]:
        """
        Get the current simulation state (playing, paused, stopped).

        Returns:
            Dict with simulation state, time, and step count.
        """
        bridge_result = await self._execute_bridge_action("get_simulation_state")
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent("""\
            import json
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                is_playing = timeline.is_playing()
                is_stopped = timeline.is_stopped()
                current_time = timeline.get_current_time()
                tps = timeline.get_time_codes_per_seconds()
                if is_playing:
                    state = "playing"
                elif is_stopped:
                    state = "stopped"
                else:
                    state = "paused"
                print(json.dumps({
                    "state": state,
                    "current_time": current_time,
                    "time_codes_per_second": tps,
                    "is_playing": is_playing,
                    "is_stopped": is_stopped,
                }))
            except Exception as e:
                print(json.dumps({"error": f"Failed to get simulation state: {e}"}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    async def start_isaac_simulation(self) -> Dict[str, Any]:
        """
        Start (play) the simulation.

        Returns:
            Dict confirming simulation started.
        """
        bridge_result = await self._execute_bridge_action(
            "simulation_control",
            {"command": "start"},
        )
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent("""\
            import json
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                timeline.play()
                print(json.dumps({"state": "playing", "started": True}))
            except Exception as e:
                print(json.dumps({"error": f"Failed to start simulation: {e}"}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    async def stop_isaac_simulation(self) -> Dict[str, Any]:
        """
        Stop the simulation and reset to initial state.

        Returns:
            Dict confirming simulation stopped.
        """
        bridge_result = await self._execute_bridge_action(
            "simulation_control",
            {"command": "stop"},
        )
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent("""\
            import json
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                timeline.stop()
                print(json.dumps({"state": "stopped", "stopped": True}))
            except Exception as e:
                print(json.dumps({"error": f"Failed to stop simulation: {e}"}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    async def pause_isaac_simulation(self) -> Dict[str, Any]:
        """
        Pause the running simulation.

        Returns:
            Dict confirming simulation paused.
        """
        bridge_result = await self._execute_bridge_action(
            "simulation_control",
            {"command": "pause"},
        )
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent("""\
            import json
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                timeline.pause()
                print(json.dumps({"state": "paused", "paused": True}))
            except Exception as e:
                print(json.dumps({"error": f"Failed to pause simulation: {e}"}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    async def step_isaac_simulation(
        self, num_steps: int = 1
    ) -> Dict[str, Any]:
        """
        Step the simulation forward by N physics steps.

        Args:
            num_steps: Number of simulation steps to advance.

        Returns:
            Dict with current time after stepping.
        """
        num_steps = max(1, min(num_steps, 1000))
        bridge_result = await self._execute_bridge_action(
            "simulation_control",
            {"command": "step", "num_steps": num_steps},
        )
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent(f"""\
            import json
            try:
                import omni.timeline
                import omni.kit.app

                timeline = omni.timeline.get_timeline_interface()
                if timeline.is_stopped():
                    timeline.play()
                    import asyncio
                    for _ in range(3):
                        await omni.kit.app.get_app().next_update_async()

                for _ in range({num_steps}):
                    await omni.kit.app.get_app().next_update_async()

                current_time = timeline.get_current_time()
                print(json.dumps({{
                    "steps": {num_steps},
                    "current_time": current_time,
                    "state": "playing" if timeline.is_playing() else "paused",
                }}))
            except Exception as e:
                print(json.dumps({{"error": f"Failed to step simulation: {{e}}"}}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    async def reset_isaac_simulation(self) -> Dict[str, Any]:
        """
        Reset the simulation to initial state (stop + set time to 0).

        Returns:
            Dict confirming reset.
        """
        bridge_result = await self._execute_bridge_action(
            "simulation_control",
            {"command": "reset"},
        )
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent("""\
            import json
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                timeline.stop()
                timeline.set_current_time(0)
                print(json.dumps({
                    "state": "stopped",
                    "current_time": 0.0,
                    "reset": True,
                }))
            except Exception as e:
                print(json.dumps({"error": f"Failed to reset simulation: {e}"}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    async def get_isaac_simulation_time(self) -> Dict[str, Any]:
        """
        Get current simulation time information.

        Returns:
            Dict with current time, FPS, and time codes per second.
        """
        script = textwrap.dedent("""\
            import json
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                print(json.dumps({
                    "current_time": timeline.get_current_time(),
                    "start_time": timeline.get_start_time(),
                    "end_time": timeline.get_end_time(),
                    "time_codes_per_second": timeline.get_time_codes_per_seconds(),
                    "is_playing": timeline.is_playing(),
                }))
            except Exception as e:
                print(json.dumps({"error": f"Failed to get simulation time: {e}"}))
        """)
        return await self._execute_json_script(script)

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
        _mat_path = json.dumps(material_path)
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
        _prim_path = json.dumps(prim_path)
        _mat_path = json.dumps(material_path)
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
        _mat_path = json.dumps(material_path)
        _prop_name = json.dumps(property_name)
        _val_str = json.dumps(json.dumps(value))
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
        diffuse_color: Optional[List[float]] = None,
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
        _mat_path = json.dumps(material_path)
        _shader_type = json.dumps(shader_type)
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

    # ------------------------------------------------------------------
    # Phase 8: Asset & Stage Operations
    # ------------------------------------------------------------------

    async def open_isaac_stage(self, file_path: str) -> Dict[str, Any]:
        """
        Open a USD stage file in Isaac Sim.

        Args:
            file_path: Path or URL to the USD file.

        Returns:
            Dict confirming the stage was opened.
        """
        _file_path = json.dumps(file_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd

            ctx = omni.usd.get_context()
            result, error = await ctx.open_stage_async({_file_path})
            if result:
                stage = ctx.get_stage()
                total = sum(1 for _ in stage.Traverse()) if stage else 0
                print(json.dumps({{
                    "file_path": {_file_path},
                    "opened": True,
                    "total_prims": total,
                }}))
            else:
                print(json.dumps({{"error": f"Failed to open stage: {{error}}"}}))
        """)
        return await self._execute_json_script(script)

    async def save_isaac_stage(
        self, file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save the current stage. If file_path is given, saves to that path.

        Args:
            file_path: Optional path to save to. None saves to current location.

        Returns:
            Dict confirming the stage was saved.
        """
        if file_path:
            _file_path = json.dumps(file_path)
            script = textwrap.dedent(f"""\
                import json
                import omni.usd

                ctx = omni.usd.get_context()
                result, error, saved_paths = await ctx.save_as_stage_async({_file_path})
                if result:
                    print(json.dumps({{"file_path": {_file_path}, "saved": True}}))
                else:
                    print(json.dumps({{"error": f"Failed to save stage: {{error}}"}}))
            """)
        else:
            script = textwrap.dedent("""\
                import json
                import omni.usd

                ctx = omni.usd.get_context()
                stage = ctx.get_stage()
                if stage is None:
                    print(json.dumps({"error": "No stage is currently open"}))
                else:
                    url = ctx.get_stage_url()
                    if not url:
                        print(json.dumps({"error": "Stage has no file path. Use file_path parameter."}))
                    else:
                        result, error, saved_paths = await ctx.save_stage_async()
                        if result:
                            print(json.dumps({"file_path": url, "saved": True}))
                        else:
                            print(json.dumps({"error": f"Failed to save stage: {error}"}))
            """)
        return await self._execute_json_script(script)

    async def new_isaac_stage(self) -> Dict[str, Any]:
        """
        Create a new empty stage in Isaac Sim.

        Returns:
            Dict confirming the new stage was created.
        """
        script = textwrap.dedent("""\
            import json
            import omni.usd

            ctx = omni.usd.get_context()
            result, error = await ctx.new_stage_async()
            if result:
                print(json.dumps({"created": True, "new_stage": True}))
            else:
                print(json.dumps({"error": f"Failed to create new stage: {error}"}))
        """)
        return await self._execute_json_script(script)

    async def import_isaac_asset(
        self,
        asset_path: str,
        target_path: str = "/World/ImportedAsset",
    ) -> Dict[str, Any]:
        """
        Import a USD asset into the current stage as a reference.

        Args:
            asset_path: File path or URL to the USD asset.
            target_path: USD path where the asset should be placed.

        Returns:
            Dict confirming the asset was imported.
        """
        _asset_path = json.dumps(asset_path)
        _target_path = json.dumps(target_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Sdf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.DefinePrim({_target_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Failed to create prim at: " + {_target_path}}}))
                else:
                    prim.GetReferences().AddReference({_asset_path})
                    print(json.dumps({{
                        "asset_path": {_asset_path},
                        "target_path": {_target_path},
                        "imported": True,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def add_isaac_reference(
        self,
        prim_path: str,
        reference_path: str,
    ) -> Dict[str, Any]:
        """
        Add a USD reference to an existing prim.

        Args:
            prim_path: USD path of the prim to add the reference to.
            reference_path: File path or URL to the referenced USD asset.

        Returns:
            Dict confirming the reference was added.
        """
        _prim_path = json.dumps(prim_path)
        _ref_path = json.dumps(reference_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    prim = stage.DefinePrim({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Cannot create prim: " + {_prim_path}}}))
                else:
                    prim.GetReferences().AddReference({_ref_path})
                    refs = prim.GetMetadata("references")
                    ref_count = len(refs.GetAddedOrExplicitItems()) if refs else 0
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "reference_path": {_ref_path},
                        "added": True,
                        "total_references": ref_count,
                    }}))
        """)
        return await self._execute_json_script(script)

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
        _prim_path = json.dumps(prim_path)
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
        _prim_path = json.dumps(prim_path)
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

    async def list_isaac_lights(self, root_path: str = "/") -> Dict[str, Any]:
        """
        List all light prims in the scene under a root path.

        Args:
            root_path: USD path to search under.

        Returns:
            Dict with count and list of lights with type, intensity, color.
        """
        _root_path = json.dumps(root_path)
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
                    print(json.dumps({{
                        "root_path": {_root_path},
                        "count": len(lights),
                        "lights": lights,
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
        _prim_path = json.dumps(prim_path)
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
        color: Optional[List[float]] = None,
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
            intensity: Light intensity.
            color: RGB color as [r, g, b] floats 0-1. Defaults to [1, 1, 1].
            color_temperature: Optional Kelvin color temperature (enables temperature mode).
            angle: Cone angle for DistantLight (degrees). Ignored for other types.
            texture_file: Texture file path for DomeLight (e.g. HDRI map).

        Returns:
            Dict confirming the light was created with its properties.
        """
        _prim_path = json.dumps(prim_path)
        _light_type = json.dumps(light_type)
        rgb = color or [1.0, 1.0, 1.0]
        _tex = json.dumps(texture_file) if texture_file else "None"
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
        _prim_path = json.dumps(prim_path)
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
        max_prims: int = 1000,
    ) -> Dict[str, Any]:
        """
        Get a full subtree as a flat list with depth info.

        Args:
            root_path: USD path of the subtree root.
            max_depth: Maximum depth to traverse.
            max_prims: Maximum number of prims to return.

        Returns:
            Dict with tree structure showing path, type, name, and depth.
        """
        _root_path = json.dumps(root_path)
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
                    for p in Usd.PrimRange(root):
                        path_str = str(p.GetPath())
                        depth = len(path_str.rstrip("/").split("/")) - 1 - root_depth
                        if depth > {max_depth}:
                            continue
                        if len(prims) >= {max_prims}:
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
                        "truncated": truncated,
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
        _prim_path = json.dumps(prim_path)
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
            root_path: USD path to compute stats under.

        Returns:
            Dict with total_prims, total_vertices, total_faces, etc.
        """
        _root_path = json.dumps(root_path)
        script = textwrap.dedent(f"""\
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
                    for p in Usd.PrimRange(root):
                        total_prims += 1
                        tname = p.GetTypeName()
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
        _root_path = json.dumps(root_path)
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
        _prim_path = json.dumps(prim_path)
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
        _prim_path = json.dumps(prim_path)
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
        _prim_path = json.dumps(prim_path)
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
        origin: List[float],
        direction: List[float],
        max_distance: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        Cast a ray into the scene and return the closest hit.

        Args:
            origin: Ray origin as [x, y, z].
            direction: Ray direction as [x, y, z] (will be normalized).
            max_distance: Maximum ray distance.

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
        center: List[float],
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
        _root_path = json.dumps(root_path)
        _prim_type = json.dumps(prim_type) if prim_type else "None"
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

    # ------------------------------------------------------------------
    # Extension management
    # ------------------------------------------------------------------

    async def list_isaac_extensions(
        self,
        enabled_only: bool = False,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List extensions registered in the running Isaac Sim instance.

        Args:
            enabled_only: If True, return only enabled extensions.
            search: Optional substring filter on extension ID.

        Returns:
            Dict with extensions list, each containing id and enabled status.
        """
        _enabled_only = repr(enabled_only)
        _search = repr(search)
        script = textwrap.dedent(f"""\
            import json
            import omni.kit.app

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            raw = ext_manager.get_extensions()
            extensions = []
            for ext in raw:
                ext_id = ext.get("id", "") or ext.get("name", "")
                enabled = ext.get("enabled", False)

                if {_enabled_only} and not enabled:
                    continue
                search_term = {_search}
                if search_term and search_term.lower() not in ext_id.lower():
                    continue

                extensions.append({{
                    "id": ext_id,
                    "enabled": enabled,
                    "version": ext.get("version", ""),
                }})

            extensions.sort(key=lambda e: e["id"])
            print(json.dumps({{
                "count": len(extensions),
                "extensions": extensions,
            }}))
        """)
        return await self._execute_json_script(script)

    async def enable_isaac_extension(self, extension_id: str) -> Dict[str, Any]:
        """
        Enable an extension by its ID in the running Isaac Sim instance.

        Args:
            extension_id: The extension name (e.g. "isaacsim.core.utils", "omni.physx").

        Returns:
            Dict with success status and extension info after enabling.
        """
        _ext_id = repr(extension_id)
        script = textwrap.dedent(f"""\
            import json
            import omni.kit.app

            ext_id = {_ext_id}
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_manager.set_extension_enabled_immediate(ext_id, True)

            # Verify the extension is now enabled
            found = False
            for ext in ext_manager.get_extensions():
                eid = ext.get("id", "") or ext.get("name", "")
                if eid == ext_id:
                    found = True
                    print(json.dumps({{
                        "extension_id": ext_id,
                        "enabled": ext.get("enabled", False),
                        "version": ext.get("version", ""),
                    }}))
                    break

            if not found:
                print(json.dumps({{"error": "Extension not found: " + ext_id}}))
        """)
        return await self._execute_json_script(script)

    # ------------------------------------------------------------------
    # Carb settings
    # ------------------------------------------------------------------

    async def get_carb_settings(self, keys: List[str]) -> Dict[str, Any]:
        """
        Read one or more Carbonite settings by key path.

        Args:
            keys: List of setting key paths (e.g. ["/rtx/fog/enabled"]).

        Returns:
            Dict mapping each key to its current value.
        """
        _keys = json.dumps(keys)
        script = textwrap.dedent(f"""\
            import json
            import carb.settings

            settings = carb.settings.get_settings()
            keys = {_keys}
            result = {{}}
            for key in keys:
                val = settings.get(key)
                try:
                    if hasattr(val, '__iter__') and not isinstance(val, (str, list, tuple, dict)):
                        val = list(val)
                except Exception:
                    val = str(val)
                result[key] = val
            print(json.dumps({{"settings": result}}))
        """)
        return await self._execute_json_script(script)

    async def set_carb_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write one or more Carbonite settings by key path.

        Args:
            settings: Dict mapping key paths to values
                      (e.g. {"/rtx/fog/enabled": true, "/rtx/fog/fogEndDist": 200.0}).

        Returns:
            Dict with applied settings and their verified new values.
        """
        _settings = json.dumps(settings)
        script = textwrap.dedent(f"""\
            import json
            import carb.settings

            cs = carb.settings.get_settings()
            to_set = {_settings}
            applied = {{}}
            for key, val in to_set.items():
                cs.set(key, val)
                new_val = cs.get(key)
                try:
                    if hasattr(new_val, '__iter__') and not isinstance(new_val, (str, list, tuple, dict)):
                        new_val = list(new_val)
                except Exception:
                    new_val = str(new_val)
                applied[key] = new_val
            print(json.dumps({{"applied": applied, "count": len(applied)}}))
        """)
        return await self._execute_json_script(script)

    # ------------------------------------------------------------------
    # AOV / Replicator
    # ------------------------------------------------------------------

    async def read_aovs(
        self,
        aov_names: List[str],
        camera_path: str = "/OmniverseKit_Persp",
        resolution: Optional[List[int]] = None,
        num_frames: int = 5,
    ) -> Dict[str, Any]:
        """
        Attach annotators, render frames, and return per-AOV statistics.

        Handles the full replicator pipeline in a single call: creates a
        render product, attaches annotators, steps the renderer, reads
        numpy data, computes statistics, cleans up, and returns results.

        Args:
            aov_names: AOV names to read (e.g. ["HdrColor", "DirectDiffuse"]).
                       Maximum 16 entries.
            camera_path: Camera prim path for the render product.
            resolution: [width, height] for the render product. Defaults to
                        [256, 256]. Max 3840x2160.
            num_frames: Number of renderer update steps before reading.
                        Clamped to [1, 60].

        Returns:
            Dict with per-AOV statistics (shape, dtype, min, max, mean,
            rgb_max, rgb_mean, nonzero_pixels for color AOVs).
        """
        if len(aov_names) > 16:
            return {"error": "aov_names must not exceed 16 entries", "error_type": "ValueError"}
        num_frames = max(1, min(num_frames, 60))
        res = resolution or [256, 256]
        if len(res) != 2:
            return {"error": "resolution must be [width, height]", "error_type": "ValueError"}
        res = [max(1, min(res[0], 3840)), max(1, min(res[1], 2160))]

        _aov_names = json.dumps(aov_names)
        _camera = json.dumps(camera_path)
        _res = json.dumps(res)
        script = textwrap.dedent(f"""\
            import json
            import numpy as np
            import omni.replicator.core as rep
            import omni.kit.app

            aov_names = {_aov_names}
            camera = {_camera}
            res = tuple({_res})
            num_frames = {num_frames}

            rp = rep.create.render_product(camera, res)
            annotators = {{}}
            attached = []
            attach_errors = {{}}
            for name in aov_names:
                try:
                    ann = rep.AnnotatorRegistry.get_annotator(name)
                    ann.attach([rp])
                    annotators[name] = ann
                    attached.append(name)
                except Exception as e:
                    attach_errors[name] = str(e)

            app = omni.kit.app.get_app()
            for _ in range(num_frames + 10):
                app.update()

            import math
            def _sf(v):
                f = float(v)
                return None if math.isnan(f) or math.isinf(f) else f

            results = {{}}
            for name, ann in annotators.items():
                try:
                    data = ann.get_data()
                    if isinstance(data, np.ndarray) and data.size > 0:
                        stats = {{
                            "shape": list(data.shape),
                            "dtype": str(data.dtype),
                            "min": _sf(data.min()),
                            "max": _sf(data.max()),
                            "mean": _sf(data.mean()),
                        }}
                        if data.ndim == 3 and data.shape[2] >= 3:
                            rgb = data[:, :, :3].astype(np.float32)
                            stats["rgb_max"] = [_sf(rgb[:,:,i].max()) for i in range(3)]
                            stats["rgb_mean"] = [_sf(rgb[:,:,i].mean()) for i in range(3)]
                            nonzero = int((rgb.max(axis=2) > 0.001).sum())
                            stats["nonzero_pixels"] = nonzero
                            stats["total_pixels"] = int(rgb.shape[0] * rgb.shape[1])
                        results[name] = stats
                    else:
                        results[name] = {{"error": "no data or empty array"}}
                except Exception as e:
                    results[name] = {{"error": str(e)}}

            for ann in annotators.values():
                ann.detach([rp])
            rp.destroy()

            output = {{
                "aovs": results,
                "attached": attached,
                "camera": camera,
                "resolution": list(res),
                "num_frames": num_frames,
            }}
            if attach_errors:
                output["attach_errors"] = attach_errors
            print(json.dumps(output))
        """)
        return await self._execute_json_script(script)

    async def list_aovs(self) -> Dict[str, Any]:
        """
        List all available AOV annotator names in the current session.

        Returns:
            Dict with list of available annotator names.
        """
        script = textwrap.dedent("""\
            import json
            import omni.replicator.core as rep

            names = sorted(rep.AnnotatorRegistry.get_registered_annotators())
            print(json.dumps({
                "count": len(names),
                "annotators": names,
            }))
        """)
        return await self._execute_json_script(script)

    # ------------------------------------------------------------------
    # USD schema queries
    # ------------------------------------------------------------------

    async def query_usd_typed_prims(
        self,
        type_name: str,
        attributes: Optional[List[str]] = None,
        root_path: str = "/",
        max_prims: int = 200,
    ) -> Dict[str, Any]:
        """
        Query prims by USD schema type and read specified attributes.

        Traverses the stage from root_path, finds prims matching the
        schema type, and reads the requested attributes from each.

        Args:
            type_name: USD schema type (e.g. "UsdLux.DistantLight",
                       "UsdGeom.Mesh", "UsdGeom.PointInstancer").
            attributes: Attribute names to read from each prim.
                        If None, returns prim paths only. Max 32 entries.
            root_path: Root path to start traversal from.
            max_prims: Maximum number of matching prims to return.
                       Clamped to [1, 2000]. Defaults to 200.

        Returns:
            Dict with list of matching prims and their attribute values.
        """
        if attributes and len(attributes) > 32:
            return {"error": "attributes must not exceed 32 entries", "error_type": "ValueError"}
        max_prims = max(1, min(max_prims, 2000))

        _type_name = json.dumps(type_name)
        _attributes = json.dumps(attributes)
        _root_path = json.dumps(root_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, UsdLux, UsdShade, Sdf, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                type_str = {_type_name}
                attr_names = [a for a in ({_attributes} or []) if a]
                root_path = {_root_path}
                max_prims = {max_prims}

                parts = type_str.split(".")
                schema_cls = None
                if len(parts) == 2:
                    mod_map = {{"UsdGeom": UsdGeom, "UsdLux": UsdLux, "UsdShade": UsdShade}}
                    mod = mod_map.get(parts[0])
                    if mod:
                        schema_cls = getattr(mod, parts[1], None)

                root_prim = stage.GetPrimAtPath(root_path) if root_path != "/" else stage.GetPseudoRoot()
                prims_data = []
                truncated = False

                for prim in Usd.PrimRange(root_prim):
                    if schema_cls is not None:
                        if not prim.IsA(schema_cls):
                            continue
                    elif prim.GetTypeName() != type_str:
                        continue

                    prim_info = {{"path": str(prim.GetPath()), "type": prim.GetTypeName()}}

                    if attr_names:
                        attrs = {{}}
                        typed_prim = schema_cls(prim) if schema_cls else None
                        for attr_name in attr_names:
                            val = None
                            if typed_prim:
                                cap_name = attr_name[0].upper() + attr_name[1:]
                                getter = getattr(typed_prim, f"Get{{cap_name}}Attr", None)
                                if getter:
                                    try:
                                        val = getter().Get()
                                    except Exception:
                                        pass
                            if val is None:
                                attr = prim.GetAttribute(attr_name)
                                if attr and attr.HasValue():
                                    val = attr.Get()
                                else:
                                    attr = prim.GetAttribute(f"inputs:{{attr_name}}")
                                    if attr and attr.HasValue():
                                        val = attr.Get()
                            if isinstance(val, (Gf.Vec3f, Gf.Vec3d, Gf.Vec4f, Gf.Vec4d)):
                                val = list(val)
                            elif isinstance(val, Gf.Matrix4d):
                                val = [list(val.GetRow(i)) for i in range(4)]
                            elif isinstance(val, Sdf.AssetPath):
                                val = str(val.path)
                            elif hasattr(val, '__len__') and not isinstance(val, (str, list, dict)):
                                try:
                                    val = list(val)
                                except Exception:
                                    val = str(val)
                            attrs[attr_name] = val
                        prim_info["attributes"] = attrs
                    prims_data.append(prim_info)

                    if len(prims_data) >= max_prims:
                        truncated = True
                        break

                print(json.dumps({{
                    "type_filter": type_str,
                    "root_path": root_path,
                    "count": len(prims_data),
                    "truncated": truncated,
                    "max_prims": max_prims,
                    "prims": prims_data,
                }}))
        """)
        return await self._execute_json_script(script)

    # ------------------------------------------------------------------
    # Viewport / render info
    # ------------------------------------------------------------------

    async def get_viewport_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the active viewport.

        Returns:
            Dict with viewport state: camera path, render product path,
            resolution, and viewport name.
        """
        script = textwrap.dedent("""\
            import json
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if not viewport:
                print(json.dumps({"error": "No active viewport found"}))
            else:
                info = {
                    "camera_path": str(viewport.camera_path),
                    "render_product_path": str(viewport.render_product_path),
                    "resolution": list(viewport.resolution),
                    "name": None,
                }
                try:
                    info["name"] = str(viewport.name)
                except Exception:
                    pass
                print(json.dumps(info))
        """)
        return await self._execute_json_script(script)

    async def list_render_vars(self) -> Dict[str, Any]:
        """
        List available render variable names from SyntheticData.

        Returns:
            Dict with render var templates and sensor type names.
        """
        script = textwrap.dedent("""\
            import json

            result = {}
            try:
                import omni.syntheticdata as syn
                sd = syn.SyntheticData.Get()
                templates = sorted(sd.get_registered_visualization_template_names())
                result["render_var_templates"] = templates
                result["render_var_count"] = len(templates)
                sensor_types = sorted(sd.get_sensor_type_names())
                result["sensor_types"] = sensor_types
                result["sensor_type_count"] = len(sensor_types)
            except Exception as e:
                result["syntheticdata_error"] = str(e)

            print(json.dumps(result))
        """)
        return await self._execute_json_script(script)

    # ------------------------------------------------------------------
    # OmniGraph
    # ------------------------------------------------------------------

    async def list_isaac_graphs(self) -> Dict[str, Any]:
        """
        List all OmniGraph graphs in the current Isaac Sim session.

        Returns:
            Dict with list of graphs, each containing path, evaluator,
            pipeline stage, node count, and backing type.
        """
        script = textwrap.dedent("""\
            import json
            import omni.graph.core as og

            graphs = og.get_all_graphs()
            result = []
            for g in graphs:
                try:
                    nodes = g.get_nodes()
                    result.append({
                        "path": g.get_path_to_graph(),
                        "evaluator": g.get_evaluator_name(),
                        "pipeline_stage": str(g.get_pipeline_stage()),
                        "node_count": len(nodes),
                        "backing_type": str(g.get_graph_backing_type()),
                    })
                except Exception:
                    pass
            print(json.dumps({"count": len(result), "graphs": result}))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_graph_nodes(
        self,
        graph_path: str,
        max_nodes: int = 200,
    ) -> Dict[str, Any]:
        """
        List nodes in an OmniGraph graph with their types and connections.

        Args:
            graph_path: USD path to the graph (e.g. "/World/ActionGraph").
            max_nodes: Maximum number of nodes to return. Clamped to [1, 1000].

        Returns:
            Dict with node list, each containing path, type, and
            input/output attribute names with connections.
        """
        max_nodes = max(1, min(max_nodes, 1000))
        _graph_path = json.dumps(graph_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.graph.core as og

            graph = og.get_graph_by_path({_graph_path})
            if not graph or not graph.is_valid():
                print(json.dumps({{"error": "Graph not found: " + {_graph_path}}}))
            else:
                nodes = graph.get_nodes()
                max_nodes = {max_nodes}
                truncated = len(nodes) > max_nodes
                result = []
                for node in nodes[:max_nodes]:
                    try:
                        attrs = node.get_attributes()
                        inputs = []
                        outputs = []
                        for attr in attrs:
                            name = attr.get_name()
                            connected = []
                            try:
                                for conn in attr.get_upstream_connections():
                                    connected.append(conn.get_path())
                            except Exception:
                                pass
                            entry = {{"name": name}}
                            if connected:
                                entry["connected_from"] = connected
                            if name.startswith("inputs:"):
                                inputs.append(entry)
                            elif name.startswith("outputs:"):
                                outputs.append(entry)
                        result.append({{
                            "path": node.get_prim_path(),
                            "type": node.get_type_name(),
                            "inputs": inputs,
                            "outputs": outputs,
                        }})
                    except Exception:
                        result.append({{
                            "path": node.get_prim_path(),
                            "type": node.get_type_name(),
                            "error": "failed to read attributes",
                        }})
                print(json.dumps({{
                    "graph_path": {_graph_path},
                    "count": len(result),
                    "truncated": truncated,
                    "nodes": result,
                }}))
        """)
        return await self._execute_json_script(script)

    async def create_isaac_graph_node(
        self,
        graph_path: str,
        node_path: str,
        node_type: str,
    ) -> Dict[str, Any]:
        """
        Create a node in an OmniGraph graph.

        Args:
            graph_path: USD path to the target graph.
            node_path: Path for the new node within the graph.
            node_type: OmniGraph node type ID
                       (e.g. "omni.graph.nodes.OnPlaybackTick").

        Returns:
            Dict with created node path and type.
        """
        _graph_path = json.dumps(graph_path)
        _node_path = json.dumps(node_path)
        _node_type = json.dumps(node_type)
        script = textwrap.dedent(f"""\
            import json
            import omni.graph.core as og

            graph = og.get_graph_by_path({_graph_path})
            if not graph or not graph.is_valid():
                print(json.dumps({{"error": "Graph not found: " + {_graph_path}}}))
            else:
                try:
                    node = graph.create_node({_node_path}, {_node_type}, True)
                    if node and node.is_valid():
                        print(json.dumps({{
                            "created": True,
                            "path": node.get_prim_path(),
                            "type": node.get_type_name(),
                        }}))
                    else:
                        print(json.dumps({{"error": "Failed to create node"}}))
                except Exception as e:
                    print(json.dumps({{"error": str(e)}}))
        """)
        return await self._execute_json_script(script)

    async def connect_isaac_graph_nodes(
        self,
        source_attr_path: str,
        target_attr_path: str,
    ) -> Dict[str, Any]:
        """
        Connect two OmniGraph node attributes.

        Args:
            source_attr_path: Full path to the source attribute
                              (e.g. "/World/Graph/OnTick.outputs:tick").
            target_attr_path: Full path to the target attribute
                              (e.g. "/World/Graph/MyNode.inputs:execIn").

        Returns:
            Dict with connection confirmation.
        """
        _source = json.dumps(source_attr_path)
        _target = json.dumps(target_attr_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.graph.core as og

            try:
                og.Controller.connect({_source}, {_target})
                print(json.dumps({{
                    "connected": True,
                    "source": {_source},
                    "target": {_target},
                }}))
            except Exception as e:
                print(json.dumps({{"error": str(e)}}))
        """)
        return await self._execute_json_script(script)

    async def set_isaac_graph_node_values(
        self,
        node_path: str,
        values: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Set attribute values on an OmniGraph node.

        Args:
            node_path: USD path to the node.
            values: Dict mapping attribute names to values
                    (e.g. {{"inputs:velocity": 1.5, "inputs:enabled": true}}).

        Returns:
            Dict with applied values.
        """
        _node_path = json.dumps(node_path)
        _values = json.dumps(values)
        script = textwrap.dedent(f"""\
            import json
            import omni.graph.core as og

            node_path = {_node_path}
            values = {_values}
            applied = {{}}
            errors = {{}}

            try:
                for attr_name, val in values.items():
                    try:
                        og.Controller.set(
                            og.Controller.attribute(attr_name, node_path),
                            val
                        )
                        applied[attr_name] = val
                    except Exception as e:
                        errors[attr_name] = str(e)
            except Exception as e:
                print(json.dumps({{"error": str(e)}}))
            else:
                result = {{"applied": applied, "count": len(applied)}}
                if errors:
                    result["errors"] = errors
                print(json.dumps(result))
        """)
        return await self._execute_json_script(script)

    async def list_isaac_graph_node_types(
        self,
        search: Optional[str] = None,
        max_types: int = 200,
    ) -> Dict[str, Any]:
        """
        List available OmniGraph node types.

        Args:
            search: Optional substring filter on node type name.
            max_types: Maximum number of types to return. Clamped to [1, 2000].

        Returns:
            Dict with list of registered node type names.
        """
        max_types = max(1, min(max_types, 2000))
        _search = json.dumps(search)
        script = textwrap.dedent(f"""\
            import json
            import omni.graph.core as og

            all_types = sorted(og.get_registered_nodes())
            search = {_search}
            if search:
                all_types = [t for t in all_types if search.lower() in t.lower()]
            max_types = {max_types}
            truncated = len(all_types) > max_types
            result = all_types[:max_types]
            print(json.dumps({{
                "count": len(result),
                "truncated": truncated,
                "total_registered": len(og.get_registered_nodes()),
                "node_types": result,
            }}))
        """)
        return await self._execute_json_script(script)

    async def delete_isaac_graph_node(
        self,
        graph_path: str,
        node_path: str,
    ) -> Dict[str, Any]:
        """
        Delete a node from an OmniGraph graph.

        Args:
            graph_path: USD path to the graph.
            node_path: Path of the node to delete.

        Returns:
            Dict with deletion confirmation.
        """
        _graph_path = json.dumps(graph_path)
        _node_path = json.dumps(node_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.graph.core as og

            graph = og.get_graph_by_path({_graph_path})
            if not graph or not graph.is_valid():
                print(json.dumps({{"error": "Graph not found: " + {_graph_path}}}))
            else:
                try:
                    success = graph.destroy_node({_node_path}, True)
                    print(json.dumps({{"deleted": success, "node_path": {_node_path}}}))
                except Exception as e:
                    print(json.dumps({{"error": str(e)}}))
        """)
        return await self._execute_json_script(script)

    # ------------------------------------------------------------------
    # Runtime diagnostics
    # ------------------------------------------------------------------

    async def get_runtime_info(self) -> Dict[str, Any]:
        """
        Get consolidated runtime diagnostics from the running Isaac Sim instance.

        Collects Kit app info, timeline state, physics stats, GPU/renderer
        info, and viewport state in a single call.

        Returns:
            Dict with app, timeline, physics, renderer, and viewport sections.
        """
        bridge_result = await self._execute_bridge_action("get_runtime_info")
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent("""\
            import json
            import sys
            import time as _time

            info = {}

            # Kit app info
            try:
                import omni.kit.app
                app = omni.kit.app.get_app()
                info["app"] = {
                    "version": str(app.get_build_version()),
                    "python_version": sys.version.split()[0],
                    "update_number": int(app.get_update_number()),
                }
            except Exception as e:
                info["app_error"] = str(e)

            # Timeline state
            try:
                import omni.timeline
                tl = omni.timeline.get_timeline_interface()
                info["timeline"] = {
                    "is_playing": tl.is_playing(),
                    "is_stopped": tl.is_stopped(),
                    "current_time": tl.get_current_time(),
                    "start_time": tl.get_start_time(),
                    "end_time": tl.get_end_time(),
                    "fps": tl.get_time_codes_per_second(),
                }
            except Exception as e:
                info["timeline_error"] = str(e)

            # Physics stats
            try:
                import omni.physx
                physx = omni.physx.get_physx_interface()
                stats = physx.get_physics_stats()
                info["physics"] = stats if isinstance(stats, dict) else {}
                info["physics"]["cuda_available"] = physx.is_cuda_lib_present()
            except Exception as e:
                info["physics_error"] = str(e)

            # Physics scene settings
            try:
                import carb.settings
                settings = carb.settings.get_settings()
                gpu_dynamics = settings.get("/physics/gpuDynamicsEnabled")
                physics_dt = settings.get("/persistent/simulation/defaultPhysicsDt")
                solver_type = settings.get("/persistent/physics/solverType")
                info["physics_config"] = {
                    "gpu_dynamics_enabled": gpu_dynamics,
                    "physics_dt": physics_dt,
                    "solver_type": solver_type,
                }
            except Exception as e:
                info["physics_config_error"] = str(e)

            # Renderer info
            try:
                import carb.settings
                settings = carb.settings.get_settings()
                info["renderer"] = {
                    "active_gpu": settings.get("/renderer/activeGpu"),
                    "gpu_name": settings.get("/renderer/gpuName"),
                    "hgi_driver": settings.get("/renderer/hgi/driver"),
                    "raytracing_mode": settings.get("/rtx/rendermode"),
                    "realtime_mode": settings.get("/rtx/ecoMode/enabled"),
                }
            except Exception as e:
                info["renderer_error"] = str(e)

            # Viewport info
            try:
                from omni.kit.viewport.utility import get_active_viewport
                viewport = get_active_viewport()
                if viewport:
                    info["viewport"] = {
                        "camera_path": str(viewport.camera_path),
                        "resolution": list(viewport.resolution),
                        "fps": viewport.fps if hasattr(viewport, "fps") else None,
                    }
                else:
                    info["viewport"] = {"status": "no active viewport"}
            except Exception as e:
                info["viewport_error"] = str(e)

            # Stage summary
            try:
                import omni.usd
                ctx = omni.usd.get_context()
                stage = ctx.get_stage()
                if stage:
                    info["stage"] = {
                        "url": ctx.get_stage_url(),
                        "prim_count": len(list(stage.Traverse())),
                    }
                else:
                    info["stage"] = {"status": "no stage open"}
            except Exception as e:
                info["stage_error"] = str(e)

            # Extensions summary
            try:
                import omni.kit.app
                ext_mgr = omni.kit.app.get_app().get_extension_manager()
                all_exts = ext_mgr.get_extensions()
                enabled = [e for e in all_exts if e.get("enabled")]
                info["extensions"] = {
                    "total": len(all_exts),
                    "enabled": len(enabled),
                }
            except Exception as e:
                info["extensions_error"] = str(e)

            print(json.dumps(info))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    async def get_isaac_logs(
        self,
        level: str = "warn",
        last_n: int = 50,
        source_filter: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Read recent log entries from the running Isaac Sim instance.

        Reads the Kit log file and returns the last N entries, optionally
        filtered by level and source module.

        Args:
            level: Minimum log level to return: "verbose", "info", "warn", "error".
                   Defaults to "warn" (warnings and errors only).
            last_n: Number of most recent matching entries to return. Max 500.
            source_filter: Optional source module substring filter (e.g. "physx", "omni.usd").
            search: Optional text search within log messages.

        Returns:
            Dict with log entries, counts per level, and log file path.
        """
        last_n = max(1, min(last_n, 500))
        _level = json.dumps(level.lower())
        _last_n = last_n
        _source_filter = json.dumps(source_filter)
        _search = json.dumps(search)
        script = textwrap.dedent(f"""\
            import json
            import re
            import os
            import glob

            level_filter = {_level}
            last_n = {_last_n}
            source_filter = {_source_filter}
            search_text = {_search}

            level_priority = {{"verbose": 0, "info": 1, "warn": 2, "warning": 2, "error": 3, "fatal": 4}}
            min_priority = level_priority.get(level_filter, 2)

            # Find the most recent Kit log file
            log_dir = os.path.expanduser("~/.nvidia-omniverse/logs/Kit")
            log_files = []
            for root, dirs, files in os.walk(log_dir):
                for f in files:
                    if f.endswith(".log"):
                        full = os.path.join(root, f)
                        log_files.append((os.path.getmtime(full), full))
            log_files.sort(reverse=True)
            log_path = log_files[0][1] if log_files else None

            if not log_path:
                print(json.dumps({{"error": "No Kit log file found"}}))
            else:
                pattern = re.compile(
                    r'\\[(Info|Warning|Warn|Error|Fatal|Verbose)\\]\\s*\\[([^\\]]+)\\]\\s*(.*)',
                    re.IGNORECASE
                )
                entries = []
                counts = {{"verbose": 0, "info": 0, "warn": 0, "error": 0}}

                with open(log_path, "r", errors="replace") as f:
                    for line in f:
                        m = pattern.search(line)
                        if not m:
                            continue
                        raw_level = m.group(1).lower()
                        if raw_level == "warning":
                            raw_level = "warn"
                        source = m.group(2)
                        message = m.group(3).strip()

                        if raw_level in counts:
                            counts[raw_level] += 1

                        priority = level_priority.get(raw_level, 0)
                        if priority < min_priority:
                            continue
                        if source_filter and source_filter.lower() not in source.lower():
                            continue
                        if search_text and search_text.lower() not in message.lower():
                            continue

                        timestamp = ""
                        ts_match = re.match(r'(\\d{{4}}-\\d{{2}}-\\d{{2}}T[\\d:]+Z)', line)
                        if ts_match:
                            timestamp = ts_match.group(1)

                        entries.append({{
                            "timestamp": timestamp,
                            "level": raw_level,
                            "source": source,
                            "message": message[:500],
                        }})

                tail = entries[-last_n:] if len(entries) > last_n else entries
                print(json.dumps({{
                    "log_file": log_path,
                    "total_matching": len(entries),
                    "returned": len(tail),
                    "counts": counts,
                    "entries": tail,
                }}))
        """)
        return await self._execute_json_script(script)

    async def set_isaac_log_level(self, level: str) -> Dict[str, Any]:
        """
        Set the Carbonite logging threshold level for the running Isaac Sim instance.

        Args:
            level: Log level: "verbose", "info", "warn", "error".

        Returns:
            Dict with the previous and new log level.
        """
        _level = json.dumps(level.lower())
        script = textwrap.dedent(f"""\
            import json
            import carb.logging

            logging = carb.logging.acquire_logging()
            level_map = {{
                "verbose": carb.logging.LEVEL_VERBOSE,
                "info": carb.logging.LEVEL_INFO,
                "warn": carb.logging.LEVEL_WARN,
                "error": carb.logging.LEVEL_ERROR,
            }}
            level_names = {{v: k for k, v in level_map.items()}}

            requested = {_level}
            if requested not in level_map:
                print(json.dumps({{"error": f"Invalid level: {{requested}}. Use: verbose, info, warn, error"}}))
            else:
                old_level = logging.get_level_threshold()
                old_name = level_names.get(old_level, str(old_level))
                logging.set_level_threshold(level_map[requested])
                new_level = logging.get_level_threshold()
                new_name = level_names.get(new_level, str(new_level))
                print(json.dumps({{
                    "previous_level": old_name,
                    "current_level": new_name,
                }}))
        """)
        return await self._execute_json_script(script)

    async def disable_isaac_extension(self, extension_id: str) -> Dict[str, Any]:
        """
        Disable an extension by its ID in the running Isaac Sim instance.

        Args:
            extension_id: The extension name (e.g. "isaacsim.core.utils", "omni.physx").

        Returns:
            Dict with success status and extension info after disabling.
        """
        _ext_id = repr(extension_id)
        script = textwrap.dedent(f"""\
            import json
            import omni.kit.app

            ext_id = {_ext_id}
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_manager.set_extension_enabled_immediate(ext_id, False)

            # Verify the extension is now disabled
            found = False
            for ext in ext_manager.get_extensions():
                eid = ext.get("id", "") or ext.get("name", "")
                if eid == ext_id:
                    found = True
                    print(json.dumps({{
                        "extension_id": ext_id,
                        "enabled": ext.get("enabled", False),
                        "version": ext.get("version", ""),
                    }}))
                    break

            if not found:
                print(json.dumps({{"error": "Extension not found: " + ext_id}}))
        """)
        return await self._execute_json_script(script)
