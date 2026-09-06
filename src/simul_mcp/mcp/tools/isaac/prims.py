"""Prim Manipulation tools for Isaac Sim."""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ....adapters import IsaacSocketClient, ScriptResult
from ...schemas.common import ErrorResponse
from ._shared import (
    DEFAULT_WORLD_PATH,
    STAGE_ROOT_PATH,
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


class PrimEditMixin:
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
        _prim_path = _pyval(prim_path)
        _prim_type = _pyval(prim_type)
        _attrs_str = _pyval(json.dumps(attributes)) if attributes else '"{}"'
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

    async def delete_isaac_prim(
        self, prim_path: str, allow_root_delete: bool = False
    ) -> Dict[str, Any]:
        """
        Delete a prim and its children from the current stage.

        The pseudo-root ``/`` is never deleted. ``/World`` is refused unless
        ``allow_root_delete`` is set, because removing it empties the scene.

        Args:
            prim_path: USD path of the prim to delete.
            allow_root_delete: Permit deleting ``/World``.

        Returns:
            Dict confirming deletion, or a RefusedOperation error.
        """
        normalized = prim_path.rstrip("/") or STAGE_ROOT_PATH
        if normalized == STAGE_ROOT_PATH:
            return self._refusal(
                "Refusing to delete the stage pseudo-root. Delete a child prim, or use "
                "new_isaac_stage to start over.",
                prim_path=prim_path,
            )
        if normalized == DEFAULT_WORLD_PATH and not allow_root_delete:
            return self._refusal(
                f"Refusing to delete {DEFAULT_WORLD_PATH}: it holds the whole scene. "
                "Pass allow_root_delete=true to delete it anyway.",
                prim_path=prim_path,
                override="allow_root_delete",
            )
        _prim_path = _pyval(prim_path)
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
        translation: Optional[FloatList] = None,
        rotation_euler: Optional[FloatList] = None,
        scale: Optional[FloatList] = None,
    ) -> Dict[str, Any]:
        """
        Set the transform of a prim (translation, rotation, scale).

        Args:
            prim_path: USD path of the prim.
            translation: Local position as [x, y, z] in stage units (metres
                by default; Isaac Sim stages are Z-up).
            rotation_euler: Rotation in degrees as [x, y, z], applied in XYZ
                Euler order.
            scale: Scale factors as [x, y, z]; 1.0 is unscaled.

        Returns:
            Dict with updated transform values.
        """
        _prim_path = _pyval(prim_path)
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
        _prim_path = _pyval(prim_path)
        _vis_token = _pyval("inherited" if visible else "invisible")
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
        _prim_path = _pyval(prim_path)
        _attr_name = _pyval(attribute_name)
        _val_str = _pyval(json.dumps(value))
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
        _prim_path = _pyval(prim_path)
        _new_path = _pyval(new_path)
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
        _prim_path = _pyval(prim_path)
        _new_parent = _pyval(new_parent_path)
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

