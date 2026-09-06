"""AOV / Replicator tools for Isaac Sim."""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ....adapters import IsaacSocketClient, ScriptResult
from ...schemas.common import ErrorResponse
from ._shared import (
    BULK_GEOMETRY_ATTRIBUTES,
    DEFAULT_MAX_RESULTS,
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


class RenderMixin:
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
            aov_names: AOV annotator names to read, as listed by
                list_isaac_aovs (e.g. ["HdrColor", "DirectDiffuse"]).
                Maximum 16 entries.
            camera_path: Camera prim path for the render product.
            resolution: [width, height] in pixels for the render product.
                Defaults to [256, 256]. Max 3840x2160.
            num_frames: Number of renderer update steps before reading.
                Clamped to [1, 60].

        Returns:
            Dict with per-AOV statistics (shape, dtype, min, max, mean,
            rgb_max, rgb_mean, nonzero_pixels for color AOVs) and a
            ``warnings`` list of cleanup problems that did not lose data.
        """
        if len(aov_names) > 16:
            return {"error": "aov_names must not exceed 16 entries", "error_type": "ValueError"}
        num_frames = max(1, min(num_frames, 60))
        res = resolution or [256, 256]
        if len(res) != 2:
            return {"error": "resolution must be [width, height]", "error_type": "ValueError"}
        res = [max(1, min(res[0], 3840)), max(1, min(res[1], 2160))]

        _aov_names = _pyval(aov_names)
        _camera = _pyval(camera_path)
        _res = _pyval(res)
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

            # Replicator only publishes annotator data on orchestrator steps.
            # Kit 110 (Isaac Sim 6.0) stopped flushing them on plain app
            # updates, so pumping frames leaves every annotator empty; the
            # explicit step also works on Kit 107 (Isaac Sim 5.1). Reading AOVs
            # must not disturb playback, so hold the timeline still:
            # delta_time=0 stops the step advancing it and pause_timeline=False
            # leaves a playing timeline playing. The frame loop stays as a
            # fallback for builds without the orchestrator.
            app = omni.kit.app.get_app()
            frame_strategy = "orchestrator"
            try:
                for _ in range(num_frames):
                    await rep.orchestrator.step_async(
                        pause_timeline=False, delta_time=0.0
                    )
            except Exception as e:
                frame_strategy = f"app_update ({{e}})"
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

            # Issue #41: in Kit 107.3.3 (Isaac Sim 5.1) rep.create.render_product
            # returns a HydraTexture wrapper. ann.attach([rp]) accepts the
            # wrapper, but ann.detach([rp]) chains down to
            # SyntheticData._get_node_path which calls
            # renderProductPath.split("/") — and HydraTexture has no .split,
            # so detach raises AttributeError mid-cleanup, dropping all AOV
            # data. Detaching by the path STRING instead bypasses the
            # split call. The wrapper exposes the path under different
            # names across Kit versions (.path attribute,
            # .get_render_product_path() method, or just the bare string),
            # so probe each form, validate the result is actually a string,
            # and only fall back to rp itself when the bare string-return
            # path is the actual return shape.
            def _extract_rp_path(rpobj):
                attr = getattr(rpobj, "path", None)
                if isinstance(attr, str):
                    return attr
                attr = getattr(rpobj, "render_product_path", None)
                if isinstance(attr, str):
                    return attr
                method = getattr(rpobj, "get_render_product_path", None)
                if callable(method):
                    try:
                        result = method()
                        if isinstance(result, str):
                            return result
                    except Exception:
                        pass
                if isinstance(rpobj, str):
                    return rpobj
                return None

            # Cleanup problems are collected, never printed on their own: the
            # caller parses stdout as one JSON object, and a second object
            # would turn the whole AOV result into a parse error.
            warnings = []
            rp_path = _extract_rp_path(rp)
            for ann in annotators.values():
                try:
                    if rp_path is not None:
                        ann.detach([rp_path])
                    else:
                        # Probing failed — say so, so a future Kit release
                        # that changes the wrapper API surfaces the issue
                        # rather than silently corrupting the cleanup.
                        warnings.append(
                            "Could not extract render-product path string from "
                            + type(rp).__name__
                            + "; detach skipped to preserve AOV data."
                        )
                except Exception as _det_err:
                    warnings.append(
                        "Detach raised "
                        + type(_det_err).__name__
                        + ": "
                        + str(_det_err)
                        + " — preserving AOV data."
                    )
            try:
                rp.destroy()
            except Exception as _destroy_err:
                warnings.append(
                    "Render product destroy raised "
                    + type(_destroy_err).__name__
                    + ": "
                    + str(_destroy_err)
                )

            output = {{
                "frame_strategy": frame_strategy,
                "aovs": results,
                "attached": attached,
                "camera": camera,
                "resolution": list(res),
                "num_frames": num_frames,
                "warnings": warnings,
            }}
            if attach_errors:
                output["attach_errors"] = attach_errors
            print(json.dumps(output))
        """)
        return await self._execute_json_script(script)

    async def list_aovs(self, max_results: int = DEFAULT_MAX_RESULTS) -> Dict[str, Any]:
        """
        List all available AOV annotator names in the current session.

        Args:
            max_results: Maximum annotator names returned. ``total`` still
                counts every registered annotator.

        Returns:
            Dict with list of available annotator names.
        """
        max_results = max(1, min(max_results, 1000))
        script = textwrap.dedent(f"""\
            import json
            import omni.replicator.core as rep

            max_results = {max_results}
            names = sorted(rep.AnnotatorRegistry.get_registered_annotators())
            print(json.dumps({{
                "count": min(len(names), max_results),
                "total": len(names),
                "truncated": len(names) > max_results,
                "annotators": names[:max_results],
            }}))
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
        max_results: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Query prims by USD schema type and read specified attributes.

        Traverses the stage from root_path, finds prims matching the
        schema type, and reads the requested attributes from each.

        Args:
            type_name: USD schema type (e.g. "UsdLux.DistantLight",
                "UsdGeom.Mesh", "UsdGeom.PointInstancer"), or any prim type
                name as returned by list_isaac_prims.
            attributes: Attribute names to read from each prim. If None,
                returns prim paths only. Max 32 entries.
            root_path: USD prim path to start the traversal from.
            max_results: Maximum number of matching prims to return per
                page. Clamped to [1, 2000]; the effective cap is reported
                as applied_limit.
            offset: Number of matching prims to skip before the page starts;
                pass the previous page's next_offset to continue.

        Returns:
            Dict with list of matching prims and their attribute values.
        """
        if attributes and len(attributes) > 32:
            return {"error": "attributes must not exceed 32 entries", "error_type": "ValueError"}
        max_results = max(1, min(max_results, 2000))
        offset = max(0, offset)

        _type_name = _pyval(type_name)
        _attributes = _pyval(attributes)
        _root_path = _pyval(root_path)
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
                max_results = {max_results}
                offset = {offset}

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
                matched = 0

                for prim in Usd.PrimRange(root_prim):
                    if schema_cls is not None:
                        if not prim.IsA(schema_cls):
                            continue
                    elif prim.GetTypeName() != type_str:
                        continue
                    matched += 1
                    if matched <= offset:
                        continue
                    if len(prims_data) >= max_results:
                        truncated = True
                        break

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

                print(json.dumps({{
                    "type_filter": type_str,
                    "root_path": root_path,
                    "count": len(prims_data),
                    "offset": offset,
                    "applied_limit": max_results,
                    "truncated": truncated,
                    "next_offset": offset + len(prims_data) if truncated else None,
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

    async def list_render_vars(
        self, max_results: int = DEFAULT_MAX_RESULTS
    ) -> Dict[str, Any]:
        """
        List available render variable names from SyntheticData.

        Args:
            max_results: Maximum names returned per list. The ``*_count``
                fields still count every registered name.

        Returns:
            Dict with render var templates and sensor type names.
        """
        max_results = max(1, min(max_results, 1000))
        script = textwrap.dedent(f"""\
            import json

            max_results = {max_results}
            result = {{}}
            try:
                import omni.syntheticdata as syn
                sd = syn.SyntheticData.Get()
                templates = sorted(sd.get_registered_visualization_template_names())
                result["render_var_templates"] = templates[:max_results]
                result["render_var_count"] = len(templates)
                sensor_types = sorted(sd.get_sensor_type_names())
                result["sensor_types"] = sensor_types[:max_results]
                result["sensor_type_count"] = len(sensor_types)
                result["truncated"] = max(len(templates), len(sensor_types)) > max_results
            except Exception as e:
                result["syntheticdata_error"] = str(e)

            print(json.dumps(result))
        """)
        return await self._execute_json_script(script)

