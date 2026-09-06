"""Per-aspect prim readers behind the get_prim_detail bridge action."""

from __future__ import annotations

from typing import Any

# Array attributes big enough that pulling their value is the cost being
# avoided. Gating on names rather than on isArray keeps small arrays such as
# xformOpOrder and primvars:displayColor readable; skipping those would lose
# information the caller needs and save nothing.
BULK_GEOMETRY_ATTRIBUTES = frozenset(
    {
        "points", "normals", "velocities", "accelerations",
        "faceVertexIndices", "faceVertexCounts", "holeIndices",
        "cornerIndices", "cornerSharpnesses", "creaseIndices",
        "creaseLengths", "creaseSharpnesses", "curveVertexCounts", "widths",
        "primvars:st", "primvars:normals",
        "positions", "orientations", "scales", "protoIndices",
        "invisibleIds", "ids",
    }
)

# Aspects a caller may request, in the order the MCP side lists them.
PRIM_DETAIL_ASPECTS: tuple[str, ...] = (
    "info",
    "transform",
    "ancestors",
    "relationships",
    "variants",
    "bounding_box",
    "mesh",
    "light",
    "material",
    "rigid_body",
    "collision",
    "joint",
    "mass",
    "animation",
)

# Light inputs the light aspect reports, keyed without the inputs: prefix.
LIGHT_INPUT_NAMES: tuple[str, ...] = (
    "inputs:intensity", "inputs:exposure", "inputs:color",
    "inputs:enableColorTemperature", "inputs:colorTemperature",
    "inputs:diffuse", "inputs:specular",
    "inputs:radius", "inputs:width", "inputs:height",
    "inputs:length", "inputs:angle", "inputs:softness",
    "inputs:shaping:cone:angle", "inputs:shaping:cone:softness",
    "inputs:shaping:focus",
)

# Primvar names that count as UV sets for the mesh aspect.
UV_PRIMVAR_NAMES: frozenset[str] = frozenset({"st", "UVMap", "st0", "st1"})


class PrimDetailReader:
    """Read the aspects of one prim inside Kit.

    Every reader returns the dict the matching MCP-side script prints, so a
    caller sees the same shape whether the bridge or the script path answered.
    A dict carrying ``"error"`` reports an aspect that does not apply to the
    prim (a mesh aspect on an Xform, a joint aspect on a Cube); the caller
    keeps the other aspects.
    """

    def __init__(self, stage: Any) -> None:
        """
        Bind the reader to an open stage.

        Args:
            stage: The Usd.Stage the prims belong to.
        """
        self._stage = stage

    def read(self, prim: Any, aspect: str) -> dict[str, Any]:
        """
        Read one aspect of a prim.

        Args:
            prim: A valid Usd.Prim.
            aspect: One of PRIM_DETAIL_ASPECTS.

        Returns:
            The aspect payload, or a dict with ``error`` when it does not apply.

        Raises:
            ValueError: When ``aspect`` is not a known aspect name.
        """
        if aspect == "info":
            return self.info(prim)
        if aspect == "transform":
            return self.transform(prim)
        if aspect == "ancestors":
            return self.ancestors(prim)
        if aspect == "relationships":
            return self.relationships(prim)
        if aspect == "variants":
            return self.variants(prim)
        if aspect == "bounding_box":
            return self.bounding_box(prim)
        if aspect == "mesh":
            return self.mesh(prim)
        if aspect == "light":
            return self.light(prim)
        if aspect == "material":
            return self.material(prim)
        if aspect == "rigid_body":
            return self.rigid_body(prim)
        if aspect == "collision":
            return self.collision(prim)
        if aspect == "joint":
            return self.joint(prim)
        if aspect == "mass":
            return self.mass(prim)
        if aspect == "animation":
            return self.animation(prim)
        raise ValueError(f"Unknown aspect: {aspect}")

    def info(self, prim: Any) -> dict[str, Any]:
        """Return type, flags, children, bound material, transform and attributes."""
        from pxr import UsdGeom, UsdShade

        child_types: dict[str, int] = {}
        children = [str(child.GetPath()) for child in prim.GetChildren()]
        for child in prim.GetChildren():
            child_type = child.GetTypeName() or "Typeless"
            child_types[child_type] = child_types.get(child_type, 0) + 1

        attrs: dict[str, Any] = {}
        for attr in prim.GetAttributes():
            attr_name = attr.GetName()
            try:
                # Bulk geometry only: Get() would decompress the whole array
                # out of the crate layer just for serialize_value to replace
                # it with an element count. Small arrays such as xformOpOrder
                # still come back in full — they carry information the caller
                # needs and cost nothing to read.
                if attr_name in BULK_GEOMETRY_ATTRIBUTES:
                    type_name = attr.GetTypeName()
                    if getattr(type_name, "isArray", False):
                        attrs[attr_name] = f"<array {type_name}>"
                        continue
                value = attr.Get()
                if value is not None:
                    attrs[attr_name] = self.serialize_value(value)
            except Exception:
                attrs[attr_name] = "<unreadable>"

        transform = None
        if prim.IsA(UsdGeom.Xformable):
            xformable = UsdGeom.Xformable(prim)
            matrix = xformable.ComputeLocalToWorldTransform(self._time_code())
            quat = matrix.ExtractRotation().GetQuat()
            transform = {
                "translation": list(matrix.ExtractTranslation()),
                "rotation_quat": [quat.GetReal()] + list(quat.GetImaginary()),
                "scale": self._scale_from_ops(xformable),
            }

        material_bindings = []
        try:
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if material:
                material_bindings.append(str(material.GetPath()))
        except Exception:
            pass

        purpose = None
        visibility = None
        if prim.IsA(UsdGeom.Imageable):
            imageable = UsdGeom.Imageable(prim)
            purpose = imageable.ComputePurpose()
            visibility = imageable.ComputeVisibility()

        return {
            "path": str(prim.GetPath()),
            "name": prim.GetName(),
            "type": prim.GetTypeName(),
            "is_active": prim.IsActive(),
            "is_defined": prim.IsDefined(),
            "is_instance": prim.IsInstance(),
            "purpose": purpose,
            "visibility": visibility,
            "children_count": len(children),
            "children_types": child_types,
            "children": children[:50],
            "material_bindings": material_bindings,
            "transform": transform,
            "attributes": attrs,
        }

    def transform(self, prim: Any, world_space: bool = True) -> dict[str, Any]:
        """Return translation, rotation quaternion and scale of an Xformable."""
        from pxr import UsdGeom

        prim_path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Xformable):
            return {"error": "Prim is not Xformable: " + prim_path}
        xformable = UsdGeom.Xformable(prim)
        if world_space:
            matrix = xformable.ComputeLocalToWorldTransform(self._time_code())
        else:
            matrix = xformable.GetLocalTransformation(self._time_code())
        quat = matrix.ExtractRotation().GetQuat()
        return {
            "prim_path": prim_path,
            "space": "world" if world_space else "local",
            "translation": list(matrix.ExtractTranslation()),
            "rotation_quat": [quat.GetReal()] + list(quat.GetImaginary()),
            "scale": self._scale_from_ops(xformable),
        }

    def ancestors(self, prim: Any) -> dict[str, Any]:
        """Return the chain from the first prim below the root down to ``prim``."""
        ancestors: list[dict[str, Any]] = []
        current = prim
        while current and current.GetPath() != current.GetParent().GetPath():
            ancestors.insert(
                0,
                {
                    "path": str(current.GetPath()),
                    "name": current.GetName(),
                    "type": current.GetTypeName(),
                },
            )
            current = current.GetParent()
        return {
            "prim_path": str(prim.GetPath()),
            "depth": len(ancestors),
            "ancestors": ancestors,
        }

    def relationships(self, prim: Any) -> dict[str, Any]:
        """Return material binding, references, payloads, variants and relationships."""
        from pxr import UsdShade

        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()

        references = []
        refs_meta = prim.GetMetadata("references")
        if refs_meta:
            for ref in refs_meta.GetAddedOrExplicitItems():
                references.append(
                    {
                        "asset_path": str(ref.assetPath) if ref.assetPath else None,
                        "prim_path": str(ref.primPath) if ref.primPath else None,
                    }
                )

        payloads = []
        pay_meta = prim.GetMetadata("payload")
        if pay_meta and hasattr(pay_meta, "GetAddedOrExplicitItems"):
            for pay in pay_meta.GetAddedOrExplicitItems():
                payloads.append(
                    {
                        "asset_path": str(pay.assetPath) if pay.assetPath else None,
                        "prim_path": str(pay.primPath) if pay.primPath else None,
                    }
                )

        relationships = []
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            if targets:
                relationships.append(
                    {"name": rel.GetName(), "targets": [str(t) for t in targets]}
                )

        return {
            "prim_path": str(prim.GetPath()),
            "material_binding": str(material.GetPath()) if material else None,
            "references": references,
            "payloads": payloads,
            "variant_sets": self._variant_sets(prim),
            "relationships": relationships,
        }

    def variants(self, prim: Any) -> dict[str, Any]:
        """Return variant sets with their variants and current selections."""
        variant_sets = self._variant_sets(prim)
        return {
            "prim_path": str(prim.GetPath()),
            "variant_set_count": len(variant_sets),
            "variant_sets": variant_sets,
        }

    def bounding_box(self, prim: Any) -> dict[str, Any]:
        """Return the world-space axis-aligned bounding box."""
        from pxr import UsdGeom

        prim_path = str(prim.GetPath())
        bbox_cache = UsdGeom.BBoxCache(self._time_code(), ["default", "render"])
        rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            return {
                "prim_path": prim_path,
                "empty": True,
                "error": "Bounding box is empty (prim may have no geometry)",
            }
        mn = rng.GetMin()
        mx = rng.GetMax()
        size = mx - mn
        center = (mn + mx) * 0.5
        return {
            "prim_path": prim_path,
            "min": [mn[0], mn[1], mn[2]],
            "max": [mx[0], mx[1], mx[2]],
            "size": [size[0], size[1], size[2]],
            "center": [center[0], center[1], center[2]],
        }

    def mesh(self, prim: Any) -> dict[str, Any]:
        """Return vertex/face counts and normal/UV presence of a Mesh."""
        from pxr import UsdGeom

        prim_path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh):
            return {"error": "Prim is not a Mesh: " + prim_path}
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        normals = mesh.GetNormalsAttr().Get()
        subdivision = mesh.GetSubdivisionSchemeAttr().Get()
        has_uvs = any(
            primvar.GetPrimvarName() in UV_PRIMVAR_NAMES
            for primvar in UsdGeom.PrimvarsAPI(prim).GetPrimvars()
        )
        return {
            "prim_path": prim_path,
            "vertex_count": len(points) if points else 0,
            "face_count": len(face_counts) if face_counts else 0,
            "face_vertex_count": len(face_indices) if face_indices else 0,
            "has_normals": normals is not None and len(normals) > 0,
            "has_uvs": has_uvs,
            "subdivision_scheme": str(subdivision) if subdivision else "none",
        }

    def light(self, prim: Any) -> dict[str, Any]:
        """Return the common light inputs and shadow flag of a light prim."""
        from pxr import UsdLux

        prim_path = str(prim.GetPath())
        if not prim.HasAPI(UsdLux.LightAPI):
            return {"error": "Prim is not a light: " + prim_path}
        info: dict[str, Any] = {"prim_path": prim_path, "type": prim.GetTypeName()}
        for attr_name in LIGHT_INPUT_NAMES:
            attr = prim.GetAttribute(attr_name)
            if not attr or attr.Get() is None:
                continue
            value = attr.Get()
            key = attr_name.replace("inputs:", "")
            try:
                if hasattr(value, "__len__") and not isinstance(value, str):
                    info[key] = [float(item) for item in value]
                elif isinstance(value, (int, float, bool)):
                    info[key] = float(value)
                else:
                    info[key] = str(value)
            except (TypeError, ValueError):
                info[key] = str(value)
        shadow_enable = prim.GetAttribute("inputs:shadow:enable")
        if shadow_enable and shadow_enable.Get() is not None:
            info["shadow_enabled"] = bool(shadow_enable.Get())
        return info

    def material(self, prim: Any) -> dict[str, Any]:
        """Return the surface shader and its inputs of a Material prim."""
        from pxr import UsdShade

        material_path = str(prim.GetPath())
        if not prim.IsA(UsdShade.Material):
            return {"error": "Prim is not a Material: " + material_path}
        shader, render_context = self._surface_shader(UsdShade.Material(prim))
        shader_path = None
        shader_type = None
        inputs: dict[str, Any] = {}
        if shader:
            shader_path = str(shader.GetPath())
            shader_type = self._shader_type(shader)
            for inp in shader.GetInputs():
                value = inp.Get()
                if value is None:
                    continue
                name = inp.GetBaseName()
                try:
                    if hasattr(value, "__len__") and not isinstance(value, str):
                        inputs[name] = [float(item) for item in value]
                    elif isinstance(value, (int, float)):
                        inputs[name] = float(value)
                    else:
                        inputs[name] = str(value)
                except Exception:
                    inputs[name] = str(value)
        return {
            "material_path": material_path,
            "shader_path": shader_path,
            "shader_type": shader_type,
            "render_context": render_context,
            "inputs": inputs,
        }

    def rigid_body(self, prim: Any) -> dict[str, Any]:
        """Return RigidBodyAPI state plus mass properties when MassAPI is applied."""
        from pxr import UsdPhysics

        prim_path = str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return {"error": "Prim does not have RigidBodyAPI: " + prim_path}
        body = UsdPhysics.RigidBodyAPI(prim)
        velocity = body.GetVelocityAttr().Get()
        angular_velocity = body.GetAngularVelocityAttr().Get()
        kinematic = body.GetKinematicEnabledAttr().Get()
        enabled = body.GetRigidBodyEnabledAttr().Get()

        mass = None
        center_of_mass = None
        inertia = None
        has_mass_api = prim.HasAPI(UsdPhysics.MassAPI)
        if has_mass_api:
            mass_api = UsdPhysics.MassAPI(prim)
            mass = mass_api.GetMassAttr().Get()
            com = mass_api.GetCenterOfMassAttr().Get()
            center_of_mass = list(com) if com else None
            diagonal = mass_api.GetDiagonalInertiaAttr().Get()
            inertia = list(diagonal) if diagonal else None

        return {
            "prim_path": prim_path,
            "rigid_body_enabled": enabled if enabled is not None else True,
            "is_kinematic": kinematic if kinematic is not None else False,
            "velocity": list(velocity) if velocity else [0, 0, 0],
            "angular_velocity": list(angular_velocity) if angular_velocity else [0, 0, 0],
            "has_mass_api": has_mass_api,
            "mass": mass,
            "center_of_mass": center_of_mass,
            "diagonal_inertia": inertia,
        }

    def collision(self, prim: Any) -> dict[str, Any]:
        """Return CollisionAPI state and the mesh approximation when present."""
        from pxr import UsdPhysics

        prim_path = str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            return {"error": "Prim does not have CollisionAPI: " + prim_path}
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        has_mesh_collision = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        approximation = None
        if has_mesh_collision:
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        return {
            "prim_path": prim_path,
            "collision_enabled": enabled if enabled is not None else True,
            "has_mesh_collision": has_mesh_collision,
            "approximation": approximation,
        }

    def joint(self, prim: Any) -> dict[str, Any]:
        """Return bodies, flags and limits of a physics joint."""
        from pxr import UsdPhysics

        prim_path = str(prim.GetPath())
        if not prim.IsA(UsdPhysics.Joint):
            return {"error": "Prim is not a Joint: " + prim_path}
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        enabled = joint.GetJointEnabledAttr().Get()

        limits: dict[str, Any] = {}
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute = UsdPhysics.RevoluteJoint(prim)
            limits = {
                "type": "revolute",
                "axis": revolute.GetAxisAttr().Get(),
                "lower": revolute.GetLowerLimitAttr().Get(),
                "upper": revolute.GetUpperLimitAttr().Get(),
            }
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            prismatic = UsdPhysics.PrismaticJoint(prim)
            limits = {
                "type": "prismatic",
                "axis": prismatic.GetAxisAttr().Get(),
                "lower": prismatic.GetLowerLimitAttr().Get(),
                "upper": prismatic.GetUpperLimitAttr().Get(),
            }

        return {
            "prim_path": prim_path,
            "joint_type": prim.GetTypeName(),
            "enabled": enabled if enabled is not None else True,
            "body0": [str(b) for b in body0] if body0 else [],
            "body1": [str(b) for b in body1] if body1 else [],
            "exclude_from_articulation": joint.GetExcludeFromArticulationAttr().Get(),
            "break_force": joint.GetBreakForceAttr().Get(),
            "break_torque": joint.GetBreakTorqueAttr().Get(),
            "limits": limits or None,
        }

    def mass(self, prim: Any) -> dict[str, Any]:
        """Return MassAPI properties."""
        from pxr import UsdPhysics

        prim_path = str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.MassAPI):
            return {"error": "Prim does not have MassAPI: " + prim_path}
        mass_api = UsdPhysics.MassAPI(prim)
        com = mass_api.GetCenterOfMassAttr().Get()
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        axes = mass_api.GetPrincipalAxesAttr().Get()
        return {
            "prim_path": prim_path,
            "mass": mass_api.GetMassAttr().Get(),
            "density": mass_api.GetDensityAttr().Get(),
            "center_of_mass": list(com) if com else None,
            "diagonal_inertia": list(inertia) if inertia else None,
            "principal_axes": (
                [axes.GetReal()] + list(axes.GetImaginary()) if axes else None
            ),
        }

    def animation(self, prim: Any) -> dict[str, Any]:
        """Return the time-sampled attributes of a prim."""
        animated: list[dict[str, Any]] = []
        for attr in prim.GetAttributes():
            num_samples = attr.GetNumTimeSamples()
            if num_samples <= 0:
                continue
            times = attr.GetTimeSamples()
            animated.append(
                {
                    "name": attr.GetName(),
                    "num_samples": num_samples,
                    "time_range": [float(times[0]), float(times[-1])] if times else None,
                }
            )
        return {
            "prim_path": str(prim.GetPath()),
            "is_animated": len(animated) > 0,
            "animated_attribute_count": len(animated),
            "animated_attributes": animated,
        }

    def _variant_sets(self, prim: Any) -> dict[str, Any]:
        """Return each variant set's variants and selection."""
        variant_sets = prim.GetVariantSets()
        result: dict[str, Any] = {}
        for name in variant_sets.GetNames():
            variant_set = variant_sets.GetVariantSet(name)
            result[name] = {
                "variants": variant_set.GetVariantNames(),
                "selection": variant_set.GetVariantSelection(),
            }
        return result

    @staticmethod
    def _time_code() -> Any:
        """Return the USD default time code."""
        from pxr import Usd

        return Usd.TimeCode.Default()

    @staticmethod
    def _scale_from_ops(xformable: Any) -> list[float]:
        """Return the first scale xform op's value, or unit scale."""
        from pxr import Gf, UsdGeom

        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                value = op.Get()
                if value is not None:
                    return list(Gf.Vec3d(value))
                break
        return [1.0, 1.0, 1.0]

    @staticmethod
    def _surface_shader(material: Any) -> tuple[Any, str]:
        """Return (shader, render_context) for the universal or MDL surface."""
        result = material.ComputeSurfaceSource()
        shader = result[0] if result else None
        if shader:
            return shader, ""
        result = material.ComputeSurfaceSource("mdl")
        return (result[0] if result else None), "mdl"

    @staticmethod
    def _shader_type(shader: Any) -> str | None:
        """Return the MDL sub-identifier or the shader id."""
        sub_id = shader.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier")
        if sub_id and sub_id.Get():
            return str(sub_id.Get())
        shader_id = shader.GetIdAttr().Get()
        return str(shader_id) if shader_id else None

    @staticmethod
    def serialize_value(value: Any) -> Any:
        """Serialize common USD values into JSON-safe Python objects."""
        from pxr import Gf

        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(
            value,
            (
                Gf.Vec2f, Gf.Vec2d, Gf.Vec2h, Gf.Vec2i,
                Gf.Vec3f, Gf.Vec3d, Gf.Vec3h, Gf.Vec3i,
                Gf.Vec4f, Gf.Vec4d, Gf.Vec4h, Gf.Vec4i,
            ),
        ):
            return [float(item) for item in value]
        if isinstance(value, (Gf.Quatf, Gf.Quatd, Gf.Quath)):
            return [float(value.GetReal())] + [float(item) for item in value.GetImaginary()]
        if isinstance(value, (Gf.Matrix4d, Gf.Matrix4f, Gf.Matrix3d, Gf.Matrix3f)):
            return str(type(value).__name__)
        try:
            if hasattr(value, "__len__") and not isinstance(value, str):
                if len(value) > 16:
                    return f"[{len(value)} elements]"
                return [PrimDetailReader.serialize_value(item) for item in value]
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return str(value)
