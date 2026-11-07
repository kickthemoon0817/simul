"""Prim Manipulation tools for Isaac Sim."""

import json
import textwrap
from typing import Any, Dict, Optional

from ._shared import (
    DEFAULT_WORLD_PATH,
    DEFINE_PRIM_CORE,
    JSON_VALUE_CORE,
    SET_PRIM_TRANSFORM_CORE,
    STAGE_ROOT_PATH,
    XFORM_READ_CORE,
    FloatList,
    _compose_script,
    _pyval,
)
from .._meta import tool_meta

# JSON value -> the attribute's USD type. pxr converts a list to a Gf vector
# by itself but not to a quaternion or a matrix (both fail with a bare "type
# mismatch"), which put xformOp:orient, Isaac's default rotation op, and a
# joint's localRot0/1 out of reach. Needs ``Gf``.
COERCE_ATTRIBUTE_VALUE_CORE = """\
    _QUATS = (Gf.Quatf, Gf.Quatd, Gf.Quath)
    _MATRIX_SIZES = {Gf.Matrix2d: 2, Gf.Matrix3d: 3, Gf.Matrix4d: 4}

    def _coerce_one(item, target, label):
        if target in _QUATS:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                raise ValueError(label + " takes a quaternion as [w, x, y, z] (real part first)")
            return target(float(item[0]), float(item[1]), float(item[2]), float(item[3]))
        if target in _MATRIX_SIZES:
            size = _MATRIX_SIZES[target]
            if not isinstance(item, (list, tuple)):
                raise ValueError(label + " takes a matrix as a list of rows")
            if item and all(isinstance(row, (list, tuple)) for row in item):
                if len(item) != size or any(len(row) != size for row in item):
                    raise ValueError(label + " takes " + str(size) + " rows of " + str(size) + " numbers")
                flat = [float(x) for row in item for x in row]
            else:
                flat = [float(x) for x in item]
            if len(flat) != size * size:
                raise ValueError(label + " takes " + str(size * size) + " numbers, row-major")
            return target(*flat)
        dimension = getattr(target, "dimension", None)
        if isinstance(dimension, int) and isinstance(item, (list, tuple)) and len(item) != dimension:
            raise ValueError(label + " takes " + str(dimension) + " components; got " + str(len(item)))
        return item

    def _coerce_attribute_value(attr, value):
        type_name = attr.GetTypeName()
        if type_name.isArray:
            if not isinstance(value, (list, tuple)):
                raise ValueError(str(type_name) + " is an array; pass a list")
            scalar_cls = type_name.scalarType.type.pythonClass
            items = [_coerce_one(item, scalar_cls, str(type_name.scalarType)) for item in value]
            if scalar_cls in _QUATS or scalar_cls in _MATRIX_SIZES:
                return type_name.type.pythonClass(items)
            return value
        return _coerce_one(value, type_name.type.pythonClass, str(type_name))
"""


class PrimEditMixin:
    # ------------------------------------------------------------------
    # Phase 3: Prim Manipulation
    # ------------------------------------------------------------------

    @tool_meta(
        name="create_isaac_prim",
        description=(
            "Create a new USD prim at the given path with optional type (Xform, Mesh, "
            "Sphere, Cube, Cylinder, Cone, Capsule, Plane)."
        ),
        read_only=False,
    )
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
        _attributes = _pyval(dict(attributes or {}))
        script = _compose_script(
            """\
            import json
            import omni.usd
            """,
            DEFINE_PRIM_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_define_prim(stage, {_prim_path}, {_prim_type}, {_attributes})))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="delete_isaac_prim",
        description=(
            "Delete a prim and all its children from the stage. Refuses the pseudo-root "
            "'/'. Refuses '/World' unless allow_root_delete=true, since that removes the "
            "whole scene."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
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

    @tool_meta(
        name="set_isaac_prim_transform",
        description=(
            "Set a prim's local transform (relative to its parent): translate, rotate (euler "
            "XYZ degrees), and/or scale. A matrix op (xformOp:transform) is converted to the "
            "equivalent translate/orient/scale stack first. An animated (time-sampled) op is "
            "refused unless time_code says which sample to write. The reply reads the result "
            "back: top-level values are local (space='local'), 'world' holds the world pose."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def set_isaac_prim_transform(
        self,
        prim_path: str,
        translation: Optional[FloatList] = None,
        rotation_euler: Optional[FloatList] = None,
        scale: Optional[FloatList] = None,
        time_code: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Set the local transform of a prim (translation, rotation, scale).

        Args:
            prim_path: USD path of the prim.
            translation: Local position as [x, y, z] in stage units (metres
                by default; Isaac Sim stages are Z-up), relative to the
                parent prim.
            rotation_euler: Local rotation in degrees as [x, y, z], applied
                in XYZ Euler order. Written to an existing orient or
                rotateXYZ op, otherwise to a new rotateXYZ op.
            scale: Local scale factors as [x, y, z]; 1.0 is unscaled.
            time_code: Time code (frame) to write the values at, for a prim
                whose transform ops are animated. Omit to write the default
                (static) value; an op that already has time samples is then
                refused, since the samples would hide the new value.

        Returns:
            Dict read back from the stage after the edit: ``space`` is
            ``"local"`` and the top-level ``translation``,
            ``rotation_quat`` ([w, x, y, z]), ``rotation_euler_xyz`` and
            ``scale`` are the local transform; ``world`` holds the same keys
            in world space. ``*_set`` echo the request and ``xform_op_order``
            the resulting op stack.
        """
        _prim_path = _pyval(prim_path)
        _translation = _pyval(list(translation) if translation is not None else None)
        _rotation = _pyval(list(rotation_euler) if rotation_euler is not None else None)
        _scale = _pyval(list(scale) if scale is not None else None)
        _time_code = _pyval(None if time_code is None else float(time_code))
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, Gf
            """,
            XFORM_READ_CORE,
            SET_PRIM_TRANSFORM_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_set_prim_transform(
                    stage, {_prim_path}, {_translation}, {_rotation}, {_scale}, {_time_code}
                )))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="set_isaac_prim_visibility",
        description="Show or hide a prim in the viewport.",
        read_only=False,
        destructive=True,
        idempotent=True,
    )
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

    @tool_meta(
        name="set_isaac_prim_attribute",
        description=(
            "Set an arbitrary attribute value on a prim, overwriting its current value. The "
            "attribute must already exist. The JSON value is converted to the attribute's "
            "USD type: quaternions (xformOp:orient, physics:localRot0) as [w, x, y, z] with "
            "the real part first, matrices (xformOp:transform) as a list of rows or a flat "
            "row-major list, vectors as [x, y, z], arrays as lists of those. Writes the "
            "default (static) value; the reply warns when time samples hide it."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
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
            value: Value to set, as JSON, converted to the attribute's type:
                a quaternion is [w, x, y, z] (real part first), a matrix a
                list of rows or a flat row-major list (16 numbers for 4x4),
                a vector a list of its components.

        Returns:
            Dict with the attribute's ``type``, ``value_set`` as given and
            ``value_read`` read back from the stage as JSON (an array over
            16 elements as a length summary), plus ``warning`` when time
            samples override the default value written.
        """
        _prim_path = _pyval(prim_path)
        _attr_name = _pyval(attribute_name)
        _val_str = _pyval(json.dumps(value))
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Gf, Sdf
            """,
            JSON_VALUE_CORE,
            COERCE_ATTRIBUTE_VALUE_CORE,
            f"""\
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
                        type_label = str(attr.GetTypeName())
                        failure = None
                        try:
                            if not attr.Set(_coerce_attribute_value(attr, val)):
                                failure = "USD refused the value"
                        except Exception as e:
                            # A Tf error reads "Error in '<function>' at line N
                            # in file <path> : '<message>'"; keep the message.
                            text = str(e).strip()
                            failure = text.splitlines()[-1].split(" : ", 1)[-1].strip(" '") if text else ""
                            failure = failure or type(e).__name__
                        if failure is not None:
                            print(json.dumps({{
                                "error": "Failed to set " + {_attr_name} + " (" + type_label + "): " + failure,
                                "type": type_label,
                            }}))
                        else:
                            read_back = attr.Get()
                            if attr.GetTypeName().isArray and read_back is not None and len(read_back) > 16:
                                value_read = {{"length": len(read_back), "elements_omitted": True}}
                            else:
                                value_read = _json_value(read_back)
                            result = {{
                                "prim_path": {_prim_path},
                                "attribute": {_attr_name},
                                "type": type_label,
                                "value_set": val,
                                "value_read": value_read,
                            }}
                            samples = attr.GetNumTimeSamples()
                            if samples:
                                result["warning"] = (
                                    "The attribute has " + str(samples) + " time samples, which "
                                    "override the default value written here at every time."
                                )
                            print(json.dumps(result))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="duplicate_isaac_prim",
        description="Duplicate a prim (deep copy) to a new path.",
        read_only=False,
    )
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

    @tool_meta(
        name="reparent_isaac_prim",
        description="Move a prim under a new parent in the hierarchy.",
        read_only=False,
        destructive=True,
        idempotent=True,
    )
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
