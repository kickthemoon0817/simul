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
        attrs_str = json.dumps(attributes) if attributes else "{}"
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                existing = stage.GetPrimAtPath("{prim_path}")
                if existing.IsValid():
                    print(json.dumps({{"error": "Prim already exists: {prim_path}"}}))
                else:
                    prim = stage.DefinePrim("{prim_path}", "{prim_type}")
                    if not prim.IsValid():
                        print(json.dumps({{"error": "Failed to create prim: {prim_path}"}}))
                    else:
                        attrs = json.loads('{attrs_str}')
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
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
                else:
                    edit = Sdf.BatchNamespaceEdit()
                    edit.Add(Sdf.NamespaceEdit.Remove("{prim_path}"))
                    if stage.GetRootLayer().Apply(edit):
                        print(json.dumps({{"prim_path": "{prim_path}", "deleted": True}}))
                    else:
                        print(json.dumps({{"error": "Failed to delete prim: {prim_path}"}}))
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
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
                elif not prim.IsA(UsdGeom.Xformable):
                    print(json.dumps({{"error": "Prim is not Xformable: {prim_path}"}}))
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
                        "prim_path": "{prim_path}",
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
        vis_token = "inherited" if visible else "invisible"
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
                elif not prim.IsA(UsdGeom.Imageable):
                    print(json.dumps({{"error": "Prim is not Imageable: {prim_path}"}}))
                else:
                    img = UsdGeom.Imageable(prim)
                    img.GetVisibilityAttr().Set("{vis_token}")
                    print(json.dumps({{
                        "prim_path": "{prim_path}",
                        "visibility": "{vis_token}",
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
        val_str = json.dumps(value)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
                else:
                    attr = prim.GetAttribute("{attribute_name}")
                    if not attr.IsValid():
                        print(json.dumps({{"error": "Attribute not found: {attribute_name} on {prim_path}"}}))
                    else:
                        val = json.loads('{val_str}')
                        try:
                            attr.Set(val)
                            read_back = attr.Get()
                            print(json.dumps({{
                                "prim_path": "{prim_path}",
                                "attribute": "{attribute_name}",
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
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                src = stage.GetPrimAtPath("{prim_path}")
                if not src.IsValid():
                    print(json.dumps({{"error": "Source prim not found: {prim_path}"}}))
                else:
                    dst = stage.GetPrimAtPath("{new_path}")
                    if dst.IsValid():
                        print(json.dumps({{"error": "Destination already exists: {new_path}"}}))
                    else:
                        Sdf.CopySpec(
                            stage.GetRootLayer(),
                            Sdf.Path("{prim_path}"),
                            stage.GetRootLayer(),
                            Sdf.Path("{new_path}"),
                        )
                        new_prim = stage.GetPrimAtPath("{new_path}")
                        if new_prim.IsValid():
                            print(json.dumps({{
                                "source_path": "{prim_path}",
                                "new_path": "{new_path}",
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
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
                else:
                    parent = stage.GetPrimAtPath("{new_parent_path}")
                    if not parent.IsValid():
                        print(json.dumps({{"error": "Parent not found: {new_parent_path}"}}))
                    else:
                        name = prim.GetName()
                        new_full_path = "{new_parent_path}/" + name
                        edit = Sdf.BatchNamespaceEdit()
                        edit.Add(
                            Sdf.NamespaceEdit.Reparent(
                                Sdf.Path("{prim_path}"),
                                Sdf.Path("{new_parent_path}"),
                                -1,
                            )
                        )
                        if stage.GetRootLayer().Apply(edit):
                            print(json.dumps({{
                                "old_path": "{prim_path}",
                                "new_path": new_full_path,
                                "parent": "{new_parent_path}",
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
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
                elif not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    print(json.dumps({{"error": "Prim does not have RigidBodyAPI: {prim_path}"}}))
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
                        "prim_path": "{prim_path}",
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
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                root = stage.GetPrimAtPath("{root_path}")
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
                    "root_path": "{root_path}",
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
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
                elif not prim.HasAPI(UsdPhysics.CollisionAPI):
                    print(json.dumps({{"error": "Prim does not have CollisionAPI: {prim_path}"}}))
                else:
                    col = UsdPhysics.CollisionAPI(prim)
                    enabled = col.GetCollisionEnabledAttr().Get()
                    # Check for mesh collision API
                    approx = None
                    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                        mesh_col = UsdPhysics.MeshCollisionAPI(prim)
                        approx = mesh_col.GetApproximationAttr().Get()
                    print(json.dumps({{
                        "prim_path": "{prim_path}",
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
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
                elif not prim.IsA(UsdPhysics.Joint):
                    print(json.dumps({{"error": "Prim is not a Joint: {prim_path}"}}))
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
                        "prim_path": "{prim_path}",
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
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
                elif not prim.HasAPI(UsdPhysics.MassAPI):
                    print(json.dumps({{"error": "Prim does not have MassAPI: {prim_path}"}}))
                else:
                    mass_api = UsdPhysics.MassAPI(prim)
                    mass = mass_api.GetMassAttr().Get()
                    density = mass_api.GetDensityAttr().Get()
                    com = mass_api.GetCenterOfMassAttr().Get()
                    inertia = mass_api.GetDiagonalInertiaAttr().Get()
                    principal_axes = mass_api.GetPrincipalAxesAttr().Get()
                    print(json.dumps({{
                        "prim_path": "{prim_path}",
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
        kin_str = "True" if kinematic else "False"
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
                else:
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        print(json.dumps({{"error": "RigidBodyAPI already applied: {prim_path}"}}))
                    else:
                        UsdPhysics.RigidBodyAPI.Apply(prim)
                        if {kin_str}:
                            rb = UsdPhysics.RigidBodyAPI(prim)
                            rb.GetKinematicEnabledAttr().Set(True)
                        print(json.dumps({{
                            "prim_path": "{prim_path}",
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
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
                else:
                    UsdPhysics.CollisionAPI.Apply(prim)
                    approx = "{approximation}"
                    if approx != "none":
                        UsdPhysics.MeshCollisionAPI.Apply(prim)
                        mesh_col = UsdPhysics.MeshCollisionAPI(prim)
                        mesh_col.GetApproximationAttr().Set(approx)
                    print(json.dumps({{
                        "prim_path": "{prim_path}",
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
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
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
                        "prim_path": "{prim_path}",
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
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdShade, UsdPhysics, Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                if not prim.IsValid():
                    # Create material prim
                    prim = stage.DefinePrim("{prim_path}", "Material")
                UsdPhysics.MaterialAPI.Apply(prim)
                mat = UsdPhysics.MaterialAPI(prim)
                mat.GetStaticFrictionAttr().Set({static_friction})
                mat.GetDynamicFrictionAttr().Set({dynamic_friction})
                mat.GetRestitutionAttr().Set({restitution})
                print(json.dumps({{
                    "prim_path": "{prim_path}",
                    "static_friction": {static_friction},
                    "dynamic_friction": {dynamic_friction},
                    "restitution": {restitution},
                    "physics_material_applied": True,
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
        return await self._execute_json_script(script)

    async def start_isaac_simulation(self) -> Dict[str, Any]:
        """
        Start (play) the simulation.

        Returns:
            Dict confirming simulation started.
        """
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
        return await self._execute_json_script(script)

    async def stop_isaac_simulation(self) -> Dict[str, Any]:
        """
        Stop the simulation and reset to initial state.

        Returns:
            Dict confirming simulation stopped.
        """
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
        return await self._execute_json_script(script)

    async def pause_isaac_simulation(self) -> Dict[str, Any]:
        """
        Pause the running simulation.

        Returns:
            Dict confirming simulation paused.
        """
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
        return await self._execute_json_script(script)

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
        return await self._execute_json_script(script)

    async def reset_isaac_simulation(self) -> Dict[str, Any]:
        """
        Reset the simulation to initial state (stop + set time to 0).

        Returns:
            Dict confirming reset.
        """
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
        return await self._execute_json_script(script)

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
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdShade

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{material_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {material_path}"}}))
                elif not prim.IsA(UsdShade.Material):
                    print(json.dumps({{"error": "Prim is not a Material: {material_path}"}}))
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
                        "material_path": "{material_path}",
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
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdShade

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{prim_path}")
                mat_prim = stage.GetPrimAtPath("{material_path}")
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: {prim_path}"}}))
                elif not mat_prim.IsValid() or not mat_prim.IsA(UsdShade.Material):
                    print(json.dumps({{"error": "Material not found: {material_path}"}}))
                else:
                    mat = UsdShade.Material(mat_prim)
                    UsdShade.MaterialBindingAPI(prim).Bind(mat)
                    print(json.dumps({{
                        "prim_path": "{prim_path}",
                        "material_path": "{material_path}",
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
        val_str = json.dumps(value)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdShade, Sdf, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath("{material_path}")
                if not prim.IsValid() or not prim.IsA(UsdShade.Material):
                    print(json.dumps({{"error": "Material not found: {material_path}"}}))
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
                        val = json.loads('{val_str}')
                        inp = shader_obj.GetInput("{property_name}")
                        if not inp or not inp.GetAttr().IsValid():
                            # Create the input for MDL shaders
                            if isinstance(val, list) and len(val) == 3:
                                inp = shader_obj.CreateInput("{property_name}", Sdf.ValueTypeNames.Color3f)
                            elif isinstance(val, list) and len(val) == 4:
                                inp = shader_obj.CreateInput("{property_name}", Sdf.ValueTypeNames.Float4)
                            elif isinstance(val, (int, float)):
                                inp = shader_obj.CreateInput("{property_name}", Sdf.ValueTypeNames.Float)
                            elif isinstance(val, bool):
                                inp = shader_obj.CreateInput("{property_name}", Sdf.ValueTypeNames.Bool)
                            elif isinstance(val, str):
                                inp = shader_obj.CreateInput("{property_name}", Sdf.ValueTypeNames.String)
                            else:
                                inp = shader_obj.CreateInput("{property_name}", Sdf.ValueTypeNames.Float)
                        if isinstance(val, list) and len(val) == 3:
                            val = Gf.Vec3f(*val)
                        elif isinstance(val, list) and len(val) == 4:
                            val = Gf.Vec4f(*val)
                        inp.Set(val)
                        print(json.dumps({{
                            "material_path": "{material_path}",
                            "property_name": "{property_name}",
                            "value_set": str(val),
                        }}))
        """)
        return await self._execute_json_script(script)

