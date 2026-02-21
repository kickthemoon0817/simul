"""
Isaac Sim MCP tools for Simul MCP Server.

This module provides granular Isaac Sim tools that execute specific
Python scripts inside a running Isaac Sim instance via TCP socket.
Each tool constructs a targeted script, sends it through the
IsaacSocketClient, and returns typed, structured responses.
"""

import json
import logging
import textwrap
from typing import Any, Dict, List, Optional

from ...adapters import IsaacSocketClient, ScriptResult
from ...config import Settings, get_settings
from ...logging import LoggerMixin, get_logger
from ..schemas import ErrorResponse

logger = get_logger(__name__)


class IsaacTools(LoggerMixin):
    """
    Tools for Isaac Sim runtime operations via TCP socket.

    Each public method constructs a Python script targeting Isaac Sim's
    omni.*, pxr.*, and isaacsim.* APIs, sends it through the socket
    client, and returns a parsed JSON response dict.
    """

    def __init__(
        self,
        client: IsaacSocketClient,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize Isaac Sim tools.

        Args:
            client: Pre-configured IsaacSocketClient for TCP communication.
            settings: Configuration settings.
        """
        self._client = client
        self.settings = settings or get_settings()

    async def _execute_json_script(self, script: str) -> Dict[str, Any]:
        """
        Execute a Python script in Isaac Sim and parse JSON from stdout.

        The script MUST print exactly one JSON object via json.dumps().
        This helper handles connectivity errors, timeouts, parse failures,
        and script-level exceptions uniformly.

        Args:
            script: Python source code that prints a JSON object to stdout.

        Returns:
            Parsed dict from stdout JSON, or an ErrorResponse dict.
        """
        try:
            result: ScriptResult = await self._client.execute(script)
        except ConnectionRefusedError:
            return ErrorResponse(
                error=(
                    f"Isaac Sim is not reachable at {self._client.address}. "
                    "Ensure Isaac Sim is running with the "
                    "isaacsim.code_editor.vscode extension enabled."
                ),
                error_type="ConnectionError",
            ).dict()
        except TimeoutError:
            return ErrorResponse(
                error=(
                    f"Script execution timed out after "
                    f"{self._client.timeout_seconds}s on "
                    f"{self._client.address}."
                ),
                error_type="TimeoutError",
            ).dict()
        except Exception as exc:
            return ErrorResponse(
                error=str(exc), error_type="Exception"
            ).dict()

        if not result.success:
            return ErrorResponse(
                error=result.error_value or "Script execution failed",
                error_type=result.error_name or "RuntimeError",
                details=(
                    {"traceback": result.traceback}
                    if result.traceback
                    else None
                ),
            ).dict()

        output = result.output.strip()
        if not output:
            return ErrorResponse(
                error="Script produced no output",
                error_type="EmptyOutput",
            ).dict()

        try:
            data: Dict[str, Any] = json.loads(output)
            data["success"] = True
            return data
        except json.JSONDecodeError as exc:
            return ErrorResponse(
                error=f"Failed to parse script output as JSON: {exc}",
                error_type="JSONDecodeError",
                details={"raw_output": output[:2000]},
            ).dict()

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
        return await self._execute_json_script(script)

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
            max_items: Maximum number of prims to return.

        Returns:
            Dict with list of prim entries (path, type, name, active).
        """
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                root = stage.GetPrimAtPath("{root_path}")
                if not root.IsValid():
                    print(json.dumps({{"error": "Invalid root path: {root_path}"}}))
                else:
                    prims = []
                    type_filter = "{prim_type or ''}"
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
                        "root_path": "{root_path}",
                        "type_filter": type_filter or None,
                        "count": len(prims),
                        "truncated": len(prims) >= max_n,
                        "prims": prims,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_prim_info(self, prim_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific prim in the running stage.

        Args:
            prim_path: USD path of the prim (e.g. "/World/Cube").

        Returns:
            Dict with prim type, attributes, transform, children, etc.
        """
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
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
                        "path": "{prim_path}",
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
        return await self._execute_json_script(script)

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
        world_str = "True" if world_space else "False"
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
                elif not prim.IsA(UsdGeom.Xformable):
                    print(json.dumps({{"error": "Prim is not Xformable: {prim_path}"}}))
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
                        "prim_path": "{prim_path}",
                        "space": "world" if world else "local",
                        "translation": list(t),
                        "rotation_quat": [r.GetReal()] + list(r.GetImaginary()),
                        "scale": s,
                    }}))
        """)
        return await self._execute_json_script(script)

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
        script = textwrap.dedent(f"""\
            import json, re
            import omni.usd
            from pxr import Usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                root = stage.GetPrimAtPath("{root_path}")
                if not root.IsValid():
                    print(json.dumps({{"error": "Invalid root path: {root_path}"}}))
                else:
                    matches = []
                    search_type = "{search_type}"
                    query = "{query}"
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
                        "root_path": "{root_path}",
                        "count": len(matches),
                        "truncated": len(matches) >= max_r,
                        "matches": matches,
                    }}))
        """)
        return await self._execute_json_script(script)

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
        self, camera_path: str = ""
    ) -> Dict[str, Any]:
        """
        Get active or specified camera parameters.

        Args:
            camera_path: USD path to a camera prim. If empty, uses active viewport camera.

        Returns:
            Dict with camera position, target, focal length, clipping range, etc.
        """
        cam_arg = camera_path or ""
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                cam_path = "{cam_arg}"
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
        camera_path: str = "",
    ) -> Dict[str, Any]:
        """
        Set camera position and/or look-at target.

        Args:
            position: Camera position as [x, y, z].
            target: Look-at target position as [x, y, z].
            camera_path: Path to camera prim. Empty uses active viewport camera.

        Returns:
            Dict confirming the updated camera state.
        """
        pos_str = str(position) if position else "None"
        tgt_str = str(target) if target else "None"
        script = textwrap.dedent(f"""\
            import json
            from pxr import Gf, Usd, UsdGeom

            pos = {pos_str}
            tgt = {tgt_str}

            try:
                import omni.kit.viewport.utility as vp_util
                from omni.kit.viewport.utility.camera_state import ViewportCameraState

                cam_path = "{camera_path}"
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
                    cam_path = "{camera_path}"
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
            width: Output image width in pixels.
            height: Output image height in pixels.

        Returns:
            Dict with base64 image data and dimensions.
        """
        script = textwrap.dedent(f"""\
            import json
            import base64
            import os
            import tempfile
            try:
                import omni.kit.viewport.utility as vp_util

                vp_api = vp_util.get_active_viewport()
                if vp_api is None:
                    print(json.dumps({{"error": "No active viewport found"}}))
                else:
                    tmp_path = os.path.join(tempfile.gettempdir(), "_simul_mcp_capture.png")
                    from omni.kit.viewport.utility import capture_viewport_to_file
                    capture_viewport_to_file(vp_api, tmp_path)

                    # Wait for the file to be written
                    import omni.kit.app
                    for _ in range(30):
                        await omni.kit.app.get_app().next_update_async()

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


