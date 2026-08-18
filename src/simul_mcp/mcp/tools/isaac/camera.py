"""Viewport & Camera tools for Isaac Sim."""

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


class CameraMixin:
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
        _cam_path = _pyval(camera_path or "")
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
        position: Optional[FloatList] = None,
        target: Optional[FloatList] = None,
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
        _cam_path = _pyval(camera_path or "")
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

                    # Read back the new state. A camera Kit has never driven
                    # through the viewport carries no center of interest, and
                    # target_world raises instead of returning None, so a
                    # position-only update must not depend on reading it.
                    new_pos = state.position_world
                    try:
                        new_tgt = [float(x) for x in state.target_world]
                    except Exception:
                        new_tgt = None
                    print(json.dumps({{
                        "camera_path": cam_path,
                        "position": [float(x) for x in new_pos],
                        "target": new_tgt,
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
        inline: bool = False,
    ) -> Dict[str, Any]:
        """
        Capture the active viewport to a PNG on the Isaac Sim host.

        Args:
            width: Output image width in pixels (1–MAX_CAPTURE_DIMENSION).
            height: Output image height in pixels (1–MAX_CAPTURE_DIMENSION).
            inline: Also return the image as base64 in the response. Only
                    honoured for files up to MAX_INLINE_CAPTURE_BYTES; above
                    that the path is returned alone, since a larger payload
                    overruns a client's per-result budget.

        Returns:
            Dict with the capture path, dimensions, and byte size. With
            ``inline`` and a small enough file, also ``image_base64``.
        """
        width = max(1, min(width, MAX_CAPTURE_DIMENSION))
        height = max(1, min(height, MAX_CAPTURE_DIMENSION))

        if inline:
            # Encode only below the cap, and say why when skipping. Emitting a
            # payload the client has to spill to a file is worse than an
            # honest path plus a reason.
            emit_body = textwrap.dedent(
                f"""\
                size_bytes = os.path.getsize(out_path)
                payload = {{
                    "path": out_path,
                    "width": {width},
                    "height": {height},
                    "format": "png",
                    "size_bytes": size_bytes,
                }}
                if size_bytes <= {MAX_INLINE_CAPTURE_BYTES}:
                    with open(out_path, "rb") as f:
                        payload["image_base64"] = base64.b64encode(
                            f.read()
                        ).decode("ascii")
                    payload["encoding"] = "base64"
                else:
                    payload["inline_skipped"] = (
                        "Capture is %d bytes, above the %d byte inline cap. "
                        "Read the file at 'path', or lower width/height."
                        % (size_bytes, {MAX_INLINE_CAPTURE_BYTES})
                    )
                print(json.dumps(payload))
                """
            )
        else:
            emit_body = textwrap.dedent(
                f"""\
                print(json.dumps({{
                    "path": out_path,
                    "width": {width},
                    "height": {height},
                    "format": "png",
                    "size_bytes": os.path.getsize(out_path),
                }}))
                """
            )

        emit = textwrap.indent(emit_body.rstrip(), " " * 24)
        max_retained = MAX_RETAINED_CAPTURES

        script = textwrap.dedent(f"""\
            import json
            import base64
            import os
            import tempfile
            import uuid
            try:
                import omni.kit.viewport.utility as vp_util
                import omni.kit.app

                vp_api = vp_util.get_active_viewport()
                if vp_api is None:
                    print(json.dumps({{"error": "No active viewport found"}}))
                else:
                    # Unique per capture: a fixed name makes an A/B pair
                    # overwrite itself, and the caller now keeps the file.
                    capture_dir = tempfile.gettempdir()
                    out_path = os.path.join(
                        capture_dir,
                        "simul_capture_%s.png" % uuid.uuid4().hex[:12],
                    )

                    # Reclaim earlier captures; the caller keeps the path, so
                    # nothing else ever deletes them.
                    try:
                        previous = sorted(
                            (
                                os.path.join(capture_dir, name)
                                for name in os.listdir(capture_dir)
                                if name.startswith("simul_capture_")
                                and name.endswith(".png")
                            ),
                            key=os.path.getmtime,
                            reverse=True,
                        )
                        for stale in previous[{max_retained} - 1:]:
                            try:
                                os.remove(stale)
                            except OSError:
                                pass
                    except OSError:
                        pass

                    # Set requested resolution on the viewport
                    vp_api.resolution = ({width}, {height})
                    # Let the viewport re-render at the new resolution
                    for _ in range(4):
                        await omni.kit.app.get_app().next_update_async()

                    # Kit 106+ (Isaac Sim 5.x): use schedule_capture with FileCapture
                    from omni.kit.widget.viewport.capture import FileCapture
                    capture = FileCapture(out_path)
                    vp_api.schedule_capture(capture)

                    # Wait for the file to be written
                    for _ in range(60):
                        await omni.kit.app.get_app().next_update_async()
                        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                            break

                    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
{emit}
                    else:
                        print(json.dumps({{"error": "Viewport capture failed — file not created"}}))
            except ImportError as e:
                print(json.dumps({{"error": f"Viewport capture not available: {{e}}"}}))
            except Exception as e:
                print(json.dumps({{"error": f"Viewport capture error: {{e}}"}}))
        """)
        return await self._execute_json_script(script)


