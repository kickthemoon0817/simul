"""Physics Inspection tools for Isaac Sim."""

import math
import textwrap
from typing import Any, Dict, Optional

from ...schemas.common import ErrorResponse
from ._shared import (
    APPLY_COLLISION_CORE,
    APPLY_RIGID_BODY_CORE,
    COLLISION_APPROXIMATIONS,
    SET_MASS_PROPERTIES_CORE,
    UNSET_CENTER_OF_MASS_NOTE,
    FloatList,
    _compose_script,
    _pyval,
)
from .joints import READ_JOINT_CORE
from .._meta import tool_meta

# PhysxSceneAPI attributes the scene tools read and write, as (result key,
# attribute name, Sdf value type). The types are the ones PhysxSchema
# declares, so an attribute authored where the schema is not registered
# (usd-core, a stage edited outside Kit) is the attribute Isaac Sim reads.
PHYSX_SCENE_SETTINGS: tuple[tuple[str, str, str], ...] = (
    ("time_steps_per_second", "physxScene:timeStepsPerSecond", "UInt"),
    ("solver_type", "physxScene:solverType", "Token"),
    ("min_position_iterations", "physxScene:minPositionIterationCount", "UInt"),
    ("max_position_iterations", "physxScene:maxPositionIterationCount", "UInt"),
    ("min_velocity_iterations", "physxScene:minVelocityIterationCount", "UInt"),
    ("max_velocity_iterations", "physxScene:maxVelocityIterationCount", "UInt"),
    ("enable_gpu_dynamics", "physxScene:enableGPUDynamics", "Bool"),
    ("broadphase_type", "physxScene:broadphaseType", "Token"),
    ("enable_ccd", "physxScene:enableCCD", "Bool"),
    ("enable_stabilization", "physxScene:enableStabilization", "Bool"),
    ("enable_enhanced_determinism", "physxScene:enableEnhancedDeterminism", "Bool"),
)

# Allowed tokens of the two PhysxSceneAPI token settings.
PHYSX_SOLVER_TYPES: tuple[str, ...] = ("TGS", "PGS")
PHYSX_BROADPHASE_TYPES: tuple[str, ...] = ("GPU", "MBP", "SAP")

# PhysX clamps solver iteration counts to 255.
MAX_SOLVER_ITERATIONS = 255

# Earth gravity in m/s^2; the stage-unit value follows from metersPerUnit.
EARTH_GRAVITY = 9.81

# Reads a UsdPhysics.Scene the way the physics parser interprets it, plus the
# PhysxSceneAPI settings as authored. Defines _stage_frame(stage) and
# _physics_scene_report(stage, prim); needs PHYSX_SCENE_SETTINGS and
# EARTH_GRAVITY from _physics_scene_constants().
PHYSICS_SCENE_REPORT_CORE = """\
    def _stage_frame(stage):
        up_axis = str(UsdGeom.GetStageUpAxis(stage))
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage)) or 1.0
        down = {"X": [-1.0, 0.0, 0.0], "Y": [0.0, -1.0, 0.0]}.get(up_axis, [0.0, 0.0, -1.0])
        return up_axis, meters_per_unit, down

    def _physx_value(value, type_name):
        if value is None:
            return None
        if type_name == "Bool":
            return bool(value)
        if type_name == "UInt":
            return int(value)
        return str(value)

    def _physics_scene_report(stage, prim):
        up_axis, meters_per_unit, down = _stage_frame(stage)
        scene = UsdPhysics.Scene(prim)
        direction = scene.GetGravityDirectionAttr().Get()
        magnitude = scene.GetGravityMagnitudeAttr().Get()
        # UsdPhysics: a zero direction means the stage's negative up axis, and
        # a negative magnitude (the fallback is -inf) means Earth gravity in
        # stage units. Both are reported resolved, as the simulation reads them.
        stage_default = False
        direction = [float(c) for c in direction] if direction is not None else [0.0, 0.0, 0.0]
        if not any(direction):
            direction = list(down)
            stage_default = True
        if magnitude is None or not math.isfinite(float(magnitude)) or float(magnitude) < 0:
            magnitude = EARTH_GRAVITY / meters_per_unit
            stage_default = True
        magnitude = float(magnitude)
        settings = {}
        authored = []
        for key, name, type_name in PHYSX_SCENE_SETTINGS:
            attr = prim.GetAttribute(name)
            # With PhysxSchema registered (Kit) an unauthored attribute reads
            # as the schema fallback; without it (usd-core) it reads None.
            settings[key] = _physx_value(attr.Get(), type_name) if attr.IsValid() else None
            if attr.IsValid() and attr.HasAuthoredValue():
                authored.append(key)
        rate = settings.get("time_steps_per_second")
        # GetAppliedSchemas lists registered schemas only; the type info keeps
        # the authored name even where PhysxSchema is not registered.
        applied = list(prim.GetAppliedSchemas()) + list(prim.GetPrimTypeInfo().GetAppliedAPISchemas())
        report = {
            "path": str(prim.GetPath()),
            "gravity_direction": direction,
            "gravity_magnitude": magnitude,
            "gravity_magnitude_m_per_s2": magnitude * meters_per_unit,
            "gravity_from_stage_defaults": stage_default,
            "physx_scene_api": "PhysxSceneAPI" in applied,
            "timestep_seconds": (1.0 / rate) if rate else None,
            "authored_settings": authored,
        }
        report.update(settings)
        return report
"""


def _physics_scene_constants() -> str:
    """Return the constants ``PHYSICS_SCENE_REPORT_CORE`` reads, as script source."""
    return (
        f"PHYSX_SCENE_SETTINGS = {_pyval(PHYSX_SCENE_SETTINGS)}\n"
        f"EARTH_GRAVITY = {_pyval(EARTH_GRAVITY)}\n"
    )


# A physics material does nothing until a collider binds it with purpose
# "physics": MaterialAPI on the collider itself, or an all-purpose bind (which
# replaces the visual material), is not what the physics parser reads.
# Defines _set_physics_material(stage, prim_path, material_path, values).
SET_PHYSICS_MATERIAL_CORE = """\
    PHYSICS_MATERIAL_DEFAULTS = {"static_friction": 0.5, "dynamic_friction": 0.5, "restitution": 0.0}
    MAX_LISTED = 200

    def _material_values(material_prim):
        api = UsdPhysics.MaterialAPI(material_prim)

        def _read(attr):
            value = attr.Get()
            return float(value) if value is not None else None

        return {
            "static_friction": _read(api.GetStaticFrictionAttr()),
            "dynamic_friction": _read(api.GetDynamicFrictionAttr()),
            "restitution": _read(api.GetRestitutionAttr()),
        }

    def _write_physics_material(stage, material_path, values):
        prim = stage.GetPrimAtPath(material_path)
        created = False
        if prim.IsValid():
            if prim.IsInstanceProxy():
                return None, False, "Material is inside an instance and cannot be edited: " + material_path
            if not prim.IsA(UsdShade.Material):
                return None, False, (
                    "Prim at " + material_path + " is a " + (prim.GetTypeName() or "typeless prim")
                    + ", not a Material"
                )
        else:
            parent = Sdf.Path(material_path).GetParentPath()
            if parent != Sdf.Path.absoluteRootPath and not stage.GetPrimAtPath(parent).IsValid():
                UsdGeom.Scope.Define(stage, parent)
            prim = UsdShade.Material.Define(stage, material_path).GetPrim()
            created = True
        # A material that only now gets MaterialAPI starts from the documented
        # defaults rather than the schema's zero friction.
        fresh = created or not prim.HasAPI(UsdPhysics.MaterialAPI)
        api = UsdPhysics.MaterialAPI.Apply(prim)
        for key, create in (
            ("static_friction", api.CreateStaticFrictionAttr),
            ("dynamic_friction", api.CreateDynamicFrictionAttr),
            ("restitution", api.CreateRestitutionAttr),
        ):
            value = values.get(key)
            if value is None and fresh:
                value = PHYSICS_MATERIAL_DEFAULTS[key]
            if value is not None:
                create().Set(float(value))
        return prim, created, None

    def _set_physics_material(stage, prim_path, material_path, values):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or prim.IsA(UsdShade.Material):
            if material_path is not None and material_path != prim_path:
                return {"error": (
                    "material_path is used only when prim_path is a collider to bind; "
                    + prim_path + (" is a Material" if prim.IsValid() else " does not exist")
                )}
            material_prim, created, error = _write_physics_material(stage, prim_path, values)
            if error is not None:
                return {"error": error}
            result = {
                "prim_path": prim_path,
                "material_path": prim_path,
                "material_created": created,
                "physics_material_applied": True,
                "bound_to": [],
                "bound_count": 0,
                "warning": (
                    "No collider was bound: a physics material has no effect until a collider "
                    "binds it with purpose physics. Pass the collider or asset root as "
                    "prim_path and this path as material_path."
                ),
            }
            result.update(_material_values(material_prim))
            return result
        if prim.IsPseudoRoot():
            return {"error": "prim_path must be a prim, not the stage root"}
        if prim.IsInstanceProxy():
            return {"error": "Prim is inside an instance and cannot be edited here: " + prim_path}

        colliders = []
        skipped_instances = []
        for candidate in Usd.PrimRange(prim):
            if candidate.IsInstance():
                skipped_instances.append(str(candidate.GetPath()))
            if candidate.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(candidate)
        if not colliders:
            message = (
                "No collision shapes at or under " + prim_path
                + "; give it collision with add_isaac_collision first"
            )
            if skipped_instances:
                message += "; instanced prims are not editable here: " + ", ".join(skipped_instances[:5])
            return {"error": message}

        if material_path is None:
            path = prim.GetPath()
            material_path = str(
                path.GetParentPath().AppendChild("PhysicsMaterials").AppendChild(path.name + "_PhysicsMaterial")
            )
        material_prim, created, error = _write_physics_material(stage, material_path, values)
        if error is not None:
            return {"error": error}
        material = UsdShade.Material(material_prim)
        bound = []
        replaced = {}
        for collider in colliders:
            binding = UsdShade.MaterialBindingAPI.Apply(collider)
            previous = binding.GetDirectBinding("physics").GetMaterialPath()
            if previous and previous != material_prim.GetPath():
                replaced[str(collider.GetPath())] = str(previous)
            # Purpose "physics" authors material:binding:physics only; the
            # all-purpose (visual) binding stays as it was.
            binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
            bound.append(str(collider.GetPath()))
        result = {
            "prim_path": prim_path,
            "material_path": material_path,
            "material_created": created,
            "physics_material_applied": True,
            "binding_purpose": "physics",
            "bound_to": bound[:MAX_LISTED],
            "bound_count": len(bound),
            "bound_truncated": len(bound) > MAX_LISTED,
        }
        if replaced:
            result["replaced_bindings"] = dict(list(replaced.items())[:MAX_LISTED])
        if skipped_instances:
            result["skipped_instances"] = skipped_instances[:MAX_LISTED]
            result["warning"] = (
                str(len(skipped_instances)) + " instanceable prim(s) skipped; their colliders "
                "are bound in the source asset"
            )
        result.update(_material_values(material_prim))
        return result
"""


class PhysicsMixin:
    # ------------------------------------------------------------------
    # Phase 4: Physics Inspection
    # ------------------------------------------------------------------

    @tool_meta(
        name="get_isaac_physics_scene",
        description=(
            "Get every physics scene on the stage: gravity (resolved the way the simulation "
            "reads it, in stage units and in m/s^2) and the PhysxSceneAPI solver settings "
            "authored on the prim: physics rate (timeStepsPerSecond) and timestep, solver "
            "type, position/velocity iteration clamps, GPU dynamics, broadphase, CCD, "
            "stabilization and enhanced determinism. Also reports the stage's upAxis and "
            "metersPerUnit. Change the solver settings with configure_isaac_physics_scene."
        ),
        read_only=True,
        idempotent=True,
    )
    async def get_isaac_physics_scene(self) -> Dict[str, Any]:
        """
        Get the physics scene configuration of the current stage.

        Returns:
            Dict with the stage's up axis and meters per unit, and one entry
            per physics scene with its resolved gravity and its PhysX scene
            settings. A setting is null when it is neither authored nor known
            to a registered PhysX schema; ``authored_settings`` names the ones
            authored on the prim.
        """
        script = _compose_script(
            """\
            import json
            import math
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics
            """,
            _physics_scene_constants(),
            PHYSICS_SCENE_REPORT_CORE,
            """\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            else:
                up_axis, meters_per_unit, _down = _stage_frame(stage)
                scenes = [
                    _physics_scene_report(stage, p) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)
                ]
                print(json.dumps({
                    "stage_up_axis": up_axis,
                    "stage_meters_per_unit": meters_per_unit,
                    "physics_scene_count": len(scenes),
                    "scenes": scenes,
                    "has_physics": len(scenes) > 0,
                }))
            """,
        )
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
            ``center_of_mass`` is null with ``center_of_mass_note`` when it
            is not authored.
        """
        _prim_path = _pyval(prim_path)
        _com_note = _pyval(UNSET_CENTER_OF_MASS_NOTE)
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
                    com_note = None
                    inertia = None
                    if prim.HasAPI(UsdPhysics.MassAPI):
                        mass_api = UsdPhysics.MassAPI(prim)
                        mass = mass_api.GetMassAttr().Get()
                        com_attr = mass_api.GetCenterOfMassAttr().Get()
                        # The schema fallback (-inf, -inf, -inf) means "derive
                        # it"; json.dumps spells it -Infinity, which is not JSON.
                        if com_attr is not None and all(abs(c) != float("inf") and c == c for c in com_attr):
                            com = list(com_attr)
                        else:
                            com_note = {_com_note}
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
                        "center_of_mass_note": com_note,
                        "diagonal_inertia": inertia,
                    }}))
        """)
        # Velocities come from USD, which GPU physics stops updating.
        return await self._execute_json_script(script, annotations=("usd_velocity_readback",))

    @tool_meta(
        name="list_isaac_physics_objects",
        description=(
            "List all prims with physics APIs: rigid bodies, colliders, and joints under a "
            "root path."
        ),
        read_only=True,
        idempotent=True,
    )
    async def list_isaac_physics_objects(
        self, root_path: str = "/", max_results: int = 200, offset: int = 0
    ) -> Dict[str, Any]:
        """
        List all prims with physics APIs applied in the stage.

        Args:
            root_path: USD prim path to search under.
            max_results: Maximum entries returned per list (rigid bodies,
                colliders and joints are paged independently). Clamped to
                [1, 1000]; the effective cap is reported as applied_limit.
            offset: Number of entries to skip in each list before the page
                starts; pass the previous page's next_offset to continue.

        Returns:
            Dict with paged lists of rigid bodies, colliders, and joints and
            the full count of each.
        """
        max_results = max(1, min(max_results, 10000))
        _root_path = _pyval(root_path)
        max_results = max(1, min(max_results, 1000))
        offset = max(0, offset)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath({_root_path}) if stage is not None else None
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            elif not root.IsValid():
                # Usd.PrimRange over an invalid prim is empty, which would read
                # as "no physics objects" rather than as a bad path.
                print(json.dumps({{"error": "Root path not found: " + {_root_path}}}))
            else:
                rigid_bodies = []
                colliders = []
                joints = []
                # Isaac robot assets keep their colliders under instanceable
                # prims; the default predicate stops there and counted none.
                for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
                    path = str(p.GetPath())
                    if p.HasAPI(UsdPhysics.RigidBodyAPI):
                        rigid_bodies.append({{"path": path, "type": p.GetTypeName()}})
                    if p.HasAPI(UsdPhysics.CollisionAPI):
                        colliders.append({{"path": path, "type": p.GetTypeName()}})
                    if p.IsA(UsdPhysics.Joint):
                        joints.append({{"path": path, "type": p.GetTypeName()}})
                offset = {offset}
                limit = {max_results}
                pages = {{
                    name: entries[offset:offset + limit]
                    for name, entries in (
                        ("rigid_bodies", rigid_bodies),
                        ("colliders", colliders),
                        ("joints", joints),
                    )
                }}
                longest = max(len(rigid_bodies), len(colliders), len(joints))
                truncated = offset + limit < longest
                print(json.dumps({{
                    "root_path": {_root_path},
                    "rigid_body_count": len(rigid_bodies),
                    "collider_count": len(colliders),
                    "joint_count": len(joints),
                    "offset": offset,
                    "applied_limit": limit,
                    "truncated": truncated,
                    "next_offset": offset + limit if truncated else None,
                    "rigid_bodies": pages["rigid_bodies"],
                    "colliders": pages["colliders"],
                    "joints": pages["joints"],
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
        _prim_path = _pyval(prim_path)
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
            Dict with joint type, bodies, break thresholds, limits (the
            revolute/prismatic pair and any LimitAPI axes) and drives (one
            entry per DriveAPI instance). Infinite values, UsdPhysics' "no
            limit" and "unbreakable", are null with a ``<key>_note``.
        """
        _prim_path = _pyval(prim_path)
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics
            """,
            READ_JOINT_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                else:
                    print(json.dumps(_read_joint(prim)))
            """,
        )
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
            ``center_of_mass`` is null with ``center_of_mass_note`` when it
            is not authored.
        """
        _prim_path = _pyval(prim_path)
        _com_note = _pyval(UNSET_CENTER_OF_MASS_NOTE)
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
                    # The schema fallback (-inf, -inf, -inf) means "derive it";
                    # json.dumps spells it -Infinity, which is not JSON.
                    com_set = com is not None and all(abs(c) != float("inf") and c == c for c in com)
                    inertia = mass_api.GetDiagonalInertiaAttr().Get()
                    principal_axes = mass_api.GetPrincipalAxesAttr().Get()
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "mass": mass,
                        "density": density,
                        "center_of_mass": list(com) if com_set else None,
                        "center_of_mass_note": None if com_set else {_com_note},
                        "diagonal_inertia": list(inertia) if inertia else None,
                        "principal_axes": (
                            [principal_axes.GetReal()] + list(principal_axes.GetImaginary())
                            if principal_axes else None
                        ),
                    }}))
        """)
        return await self._execute_json_script(script)

    # ------------------------------------------------------------------
    # Phase 5: Physics Configuration
    # ------------------------------------------------------------------

    @tool_meta(
        name="add_isaac_rigid_body",
        description="Apply RigidBodyAPI to a prim, optionally making it kinematic.",
        read_only=False,
        destructive=True,
    )
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
        _prim_path = _pyval(prim_path)
        _kinematic = _pyval(bool(kinematic))
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import UsdPhysics
            """,
            APPLY_RIGID_BODY_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_apply_rigid_body(stage, {_prim_path}, {_kinematic})))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="add_isaac_collision",
        description=(
            "Make a prim collide: applies CollisionAPI to the collision geometry (Mesh, Cube, "
            "Sphere, Capsule, Cylinder, Cone, Plane, Points) at and, by default, under "
            "prim_path, so an asset root Xform gets colliders on its meshes rather than an "
            "API the physics parser ignores. approximation (none, convexHull, "
            "convexDecomposition, meshSimplification, boundingSphere, boundingCube) is set on "
            "meshes through MeshCollisionAPI; implicit shapes stay analytic. Going down the "
            "tree, render-purpose (visual-only) and guide geometry is skipped and reported "
            "unless all_purposes=true. Reports every shape it applied collision to; fails "
            "when there is no geometry."
        ),
        read_only=False,
        destructive=True,
    )
    async def add_isaac_collision(
        self,
        prim_path: str,
        approximation: Optional[str] = None,
        recursive: bool = True,
        all_purposes: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply CollisionAPI to the collision geometry at or under a prim.

        Args:
            prim_path: USD path of a geometry prim, or of an asset root whose
                geometry should collide.
            approximation: MeshCollisionAPI approximation for meshes: "none"
                (the triangle mesh, static or kinematic bodies only),
                "convexHull", "convexDecomposition", "meshSimplification",
                "boundingSphere", "boundingCube". Omit to keep each mesh's
                authored approximation (UsdPhysics' default is none).
                Implicit shapes (Cube, Sphere, ...) ignore it.
            recursive: Also apply collision to the geometry under prim_path.
                With false, prim_path itself must be a geometry prim.
                Instanceable prims are skipped and reported: their geometry
                is edited in the source asset.
            all_purposes: With recursive, also give collision to geometry
                whose purpose is render (visual-only, such as a robot's
                high-detail visual meshes) or guide. Off by default; a
                prim_path given directly is always included.

        Returns:
            Dict with ``shapes`` (path, type, the approximation in effect,
            and whether it already had collision), ``shape_count`` and
            ``skipped_purpose`` for render/guide geometry left out, or an
            error when no collision geometry was found.
        """
        if approximation is not None and approximation not in COLLISION_APPROXIMATIONS:
            return ErrorResponse(
                error=(
                    f"Invalid approximation {approximation!r}. Must be one of: "
                    f"{', '.join(COLLISION_APPROXIMATIONS)}"
                ),
                error_type="ValidationError",
            ).model_dump()
        _prim_path = _pyval(prim_path)
        _approx = _pyval(approximation)
        _recursive = _pyval(bool(recursive))
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics
            """,
            APPLY_COLLISION_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_apply_collision(
                    stage, {_prim_path}, {_approx}, {_recursive}, {_pyval(bool(all_purposes))}
                )))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="set_isaac_mass_properties",
        description=(
            "Set mass, density, and/or center of mass on a prim. Applies MassAPI if not "
            "already present."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def set_isaac_mass_properties(
        self,
        prim_path: str,
        mass: Optional[float] = None,
        density: Optional[float] = None,
        center_of_mass: Optional[FloatList] = None,
    ) -> Dict[str, Any]:
        """
        Set mass properties on a prim (applies MassAPI if not present).

        Args:
            prim_path: USD path of the prim.
            mass: Mass in kg.
            density: Density in kg/m^3 (used when mass is not set).
            center_of_mass: Center of mass as [x, y, z] in the prim's local
                frame, stage units (metres by default).

        Returns:
            Dict confirming updated mass properties.
        """
        _prim_path = _pyval(prim_path)
        _mass = _pyval(mass)
        _density = _pyval(density)
        _center_of_mass = _pyval(list(center_of_mass) if center_of_mass else None)
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import UsdPhysics, Gf
            """,
            SET_MASS_PROPERTIES_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_set_mass_properties(
                    stage, {_prim_path}, {_mass}, {_density}, {_center_of_mass}
                )))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="set_isaac_physics_material",
        description=(
            "Give colliders friction and restitution. Pass the collider, or an asset root, as "
            "prim_path: a physics material (UsdPhysics.MaterialAPI on a Material prim, at "
            "material_path or <parent>/PhysicsMaterials/<name>_PhysicsMaterial, created or "
            "reused) is bound with purpose 'physics' to every collision shape at or under it; "
            "the visual material binding is left alone. Reports the colliders bound. A "
            "Material prim, or a path where nothing exists, as prim_path only creates or "
            "updates that material and binds nothing (reported in a warning)."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def set_isaac_physics_material(
        self,
        prim_path: str,
        static_friction: Optional[float] = None,
        dynamic_friction: Optional[float] = None,
        restitution: Optional[float] = None,
        material_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create or update a physics material and bind it to colliders.

        Args:
            prim_path: The collider, or a prim whose descendants collide, to
                bind the material to. A Material prim, or a path where no
                prim exists, is the material itself: it is created or
                updated and nothing is bound.
            static_friction: Static friction coefficient (>= 0). A new
                material defaults to 0.5; an existing one keeps its value.
            dynamic_friction: Dynamic friction coefficient (>= 0). A new
                material defaults to 0.5; an existing one keeps its value.
            restitution: Restitution (bounciness) in [0, 1]. A new material
                defaults to 0.0; an existing one keeps its value.
            material_path: Material prim to create or reuse when prim_path is
                a collider. Defaults to
                "<parent>/PhysicsMaterials/<name>_PhysicsMaterial".

        Returns:
            Dict with ``material_path``, ``material_created``, the friction
            and restitution now on the material, ``bound_to`` (the colliders
            now bound with purpose physics) and ``replaced_bindings`` for
            colliders that were bound to another physics material.
        """
        for name, value in (
            ("static_friction", static_friction),
            ("dynamic_friction", dynamic_friction),
        ):
            if value is not None and not value >= 0:
                return ErrorResponse(
                    error=f"{name} must be >= 0, got {value!r}", error_type="ValidationError"
                ).model_dump()
        if restitution is not None and not 0.0 <= restitution <= 1.0:
            return ErrorResponse(
                error=f"restitution must be in [0, 1], got {restitution!r}",
                error_type="ValidationError",
            ).model_dump()
        values = {
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
            "restitution": restitution,
        }
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
            """,
            SET_PHYSICS_MATERIAL_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_set_physics_material(
                    stage, {_pyval(prim_path)}, {_pyval(material_path)}, {_pyval(values)}
                )))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="create_isaac_physics_scene",
        description=(
            "Create a UsdPhysics.Scene prim, or update the gravity of the scene already at "
            "prim_path (reported as updated=true). A new scene gets Earth gravity along the "
            "stage's negative up axis, scaled to its metersPerUnit (a Y-up centimetre stage "
            "gets [0, -1, 0] at 981 units/s^2); gravity_magnitude is taken in m/s^2 and "
            "converted the same way. Any other prim at prim_path is refused, never retyped. "
            "This is the prerequisite for any physics simulation — rigid bodies, colliders, "
            "and joints require a physics scene to function. Solver settings: "
            "configure_isaac_physics_scene."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def create_isaac_physics_scene(
        self,
        prim_path: str = "/World/PhysicsScene",
        gravity_direction: Optional[FloatList] = None,
        gravity_magnitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Create a UsdPhysics.Scene prim, or update the gravity of an existing one.

        Args:
            prim_path: USD path for the physics scene prim. A prim of another
                type already there is refused rather than retyped.
            gravity_direction: Gravity direction [x, y, z] in stage axes,
                normalised before it is written. A new scene defaults to the
                stage's negative up axis ([0, 0, -1] on a Z-up stage,
                [0, -1, 0] on a Y-up one); an existing scene keeps its
                direction unless one is passed.
            gravity_magnitude: Gravity magnitude in m/s^2 (>= 0, along
                gravity_direction), written in stage units: divided by the
                stage's metersPerUnit, so 9.81 becomes 981 on a centimetre
                stage. A new scene defaults to 9.81; an existing scene keeps
                its magnitude unless one is passed.

        Returns:
            Dict with the scene's gravity in stage units and in m/s^2, the
            stage's up axis and meters per unit, ``created`` for a new prim,
            and ``already_existed`` plus ``updated`` for a scene that was
            there; an error for a prim of another type at prim_path.
        """
        direction: Optional[list[float]] = None
        if gravity_direction is not None:
            components = [float(component) for component in gravity_direction]
            length = math.sqrt(sum(c * c for c in components)) if len(components) == 3 else 0.0
            if len(components) != 3 or not math.isfinite(length) or length == 0.0:
                return ErrorResponse(
                    error=(
                        "gravity_direction must be three finite numbers, not all zero; got "
                        f"{list(gravity_direction)!r}"
                    ),
                    error_type="ValidationError",
                ).model_dump()
            direction = [c / length for c in components]
        if gravity_magnitude is not None and not (
            math.isfinite(gravity_magnitude) and gravity_magnitude >= 0
        ):
            return ErrorResponse(
                error=f"gravity_magnitude must be a finite value >= 0 (m/s^2), got {gravity_magnitude!r}",
                error_type="ValidationError",
            ).model_dump()
        magnitude = float(gravity_magnitude) if gravity_magnitude is not None else None
        script = _compose_script(
            """\
            import json
            import math
            import omni.usd
            from pxr import Gf, Usd, UsdGeom, UsdPhysics
            """,
            _physics_scene_constants(),
            PHYSICS_SCENE_REPORT_CORE,
            f"""\
            prim_path = {_pyval(prim_path)}
            requested_direction = {_pyval(direction)}
            requested_magnitude = {_pyval(magnitude)}
            """,
            """\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            else:
                up_axis, meters_per_unit, down = _stage_frame(stage)
                existing = stage.GetPrimAtPath(prim_path)
                scene = None
                if existing.IsValid() and not existing.IsA(UsdPhysics.Scene):
                    # DefinePrim would retype it: an Xform robot root would
                    # become a scene that can no longer be transformed.
                    result = {
                        "error": (
                            "A " + (existing.GetTypeName() or "typeless prim") + " already exists at "
                            + prim_path + "; refusing to turn it into a PhysicsScene. "
                            "Pass a prim_path that is free or already a PhysicsScene."
                        ),
                        "error_type": "RefusedOperation",
                        "prim_path": prim_path,
                        "existing_type": existing.GetTypeName(),
                    }
                elif existing.IsValid():
                    scene = UsdPhysics.Scene(existing)
                    updated = False
                    if requested_direction is not None:
                        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(*requested_direction))
                        updated = True
                    if requested_magnitude is not None:
                        scene.CreateGravityMagnitudeAttr().Set(requested_magnitude / meters_per_unit)
                        updated = True
                    result = {
                        "prim_path": prim_path,
                        "already_existed": True,
                        "created": False,
                        "updated": updated,
                        "message": (
                            "Physics scene already exists at this path; gravity updated"
                            if updated
                            else "Physics scene already exists at this path; gravity unchanged"
                        ),
                    }
                else:
                    scene = UsdPhysics.Scene.Define(stage, prim_path)
                    if not scene:
                        result = {"error": "Failed to create physics scene prim at " + prim_path}
                        scene = None
                    else:
                        direction = requested_direction if requested_direction is not None else down
                        magnitude = requested_magnitude if requested_magnitude is not None else EARTH_GRAVITY
                        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(*direction))
                        scene.CreateGravityMagnitudeAttr().Set(magnitude / meters_per_unit)
                        result = {
                            "prim_path": prim_path,
                            "already_existed": False,
                            "created": True,
                            "updated": False,
                            "gravity_source": (
                                "requested"
                                if requested_direction is not None and requested_magnitude is not None
                                else "stage up axis and metersPerUnit"
                            ),
                        }
                if scene is not None:
                    report = _physics_scene_report(stage, scene.GetPrim())
                    for key in ("gravity_direction", "gravity_magnitude", "gravity_magnitude_m_per_s2"):
                        result[key] = report[key]
                    result["stage_up_axis"] = up_axis
                    result["stage_meters_per_unit"] = meters_per_unit
                print(json.dumps(result))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="configure_isaac_physics_scene",
        description=(
            "Set PhysX solver settings on a physics scene: physics rate "
            "(time_steps_per_second; timestep = 1/rate), solver_type (TGS, PGS), the "
            "position/velocity solver iteration clamps (min/max, up to 255), "
            "enable_gpu_dynamics, broadphase_type (GPU, MBP, SAP), enable_ccd, "
            "enable_stabilization and enable_enhanced_determinism. Applies PhysxSceneAPI and "
            "authors only the settings passed, then returns every setting as it now reads. "
            "prim_path defaults to the stage's only physics scene."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def configure_isaac_physics_scene(
        self,
        prim_path: Optional[str] = None,
        time_steps_per_second: Optional[int] = None,
        solver_type: Optional[str] = None,
        min_position_iterations: Optional[int] = None,
        max_position_iterations: Optional[int] = None,
        min_velocity_iterations: Optional[int] = None,
        max_velocity_iterations: Optional[int] = None,
        enable_gpu_dynamics: Optional[bool] = None,
        broadphase_type: Optional[str] = None,
        enable_ccd: Optional[bool] = None,
        enable_stabilization: Optional[bool] = None,
        enable_enhanced_determinism: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Author PhysxSceneAPI solver settings on a physics scene.

        Args:
            prim_path: The PhysicsScene prim. Defaults to the only physics
                scene on the stage; required when there are several.
            time_steps_per_second: Physics steps per simulated second (>= 1);
                the physics timestep is 1 / this.
            solver_type: "TGS" (temporal Gauss-Seidel) or "PGS"
                (projected Gauss-Seidel).
            min_position_iterations: Lower clamp on any body's position
                solver iterations, 1-255.
            max_position_iterations: Upper clamp on position iterations,
                1-255.
            min_velocity_iterations: Lower clamp on velocity iterations,
                0-255.
            max_velocity_iterations: Upper clamp on velocity iterations,
                0-255.
            enable_gpu_dynamics: Run rigid-body dynamics on the GPU.
            broadphase_type: "GPU", "MBP" (multi box pruning) or "SAP"
                (sweep and prune).
            enable_ccd: Enable continuous collision detection in the scene.
            enable_stabilization: Enable PhysX stabilization.
            enable_enhanced_determinism: Enable enhanced determinism, which
                keeps results reproducible across runs at some cost.

        Returns:
            Dict with ``applied`` (the settings written) and the scene's
            settings as ``get_isaac_physics_scene`` reports them.
        """
        requested: Dict[str, Any] = {
            key: value
            for key, value in (
                ("time_steps_per_second", time_steps_per_second),
                ("solver_type", solver_type.upper() if solver_type is not None else None),
                ("min_position_iterations", min_position_iterations),
                ("max_position_iterations", max_position_iterations),
                ("min_velocity_iterations", min_velocity_iterations),
                ("max_velocity_iterations", max_velocity_iterations),
                ("enable_gpu_dynamics", enable_gpu_dynamics),
                ("broadphase_type", broadphase_type.upper() if broadphase_type is not None else None),
                ("enable_ccd", enable_ccd),
                ("enable_stabilization", enable_stabilization),
                ("enable_enhanced_determinism", enable_enhanced_determinism),
            )
            if value is not None
        }
        problem = _physx_settings_problem(requested)
        if problem is not None:
            return ErrorResponse(error=problem, error_type="ValidationError").model_dump()
        script = _compose_script(
            """\
            import json
            import math
            import omni.usd
            from pxr import Sdf, Usd, UsdGeom, UsdPhysics
            """,
            _physics_scene_constants(),
            PHYSICS_SCENE_REPORT_CORE,
            f"""\
            prim_path = {_pyval(prim_path)}
            requested = {_pyval(requested)}
            """,
            """\
            stage = omni.usd.get_context().get_stage()
            prim = None
            if stage is None:
                result = {"error": "No stage is currently open"}
            elif prim_path is None:
                scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
                if not scenes:
                    result = {"error": "No physics scene on the stage; create one with create_isaac_physics_scene"}
                elif len(scenes) > 1:
                    result = {
                        "error": str(len(scenes)) + " physics scenes on the stage; pass prim_path",
                        "scenes": [str(p.GetPath()) for p in scenes],
                    }
                else:
                    prim = scenes[0]
            else:
                candidate = stage.GetPrimAtPath(prim_path)
                if not candidate.IsValid():
                    result = {"error": "Prim not found: " + prim_path}
                elif not candidate.IsA(UsdPhysics.Scene):
                    result = {"error": "Prim is not a PhysicsScene: " + prim_path}
                else:
                    prim = candidate
            if prim is not None:
                merged = _physics_scene_report(stage, prim)
                merged.update(requested)
                for kind in ("position", "velocity"):
                    low = merged.get("min_" + kind + "_iterations")
                    high = merged.get("max_" + kind + "_iterations")
                    if low is not None and high is not None and low > high:
                        result = {"error": (
                            "min_" + kind + "_iterations (" + str(low) + ") would exceed max_"
                            + kind + "_iterations (" + str(high) + "); nothing was changed"
                        )}
                        prim = None
                        break
            if prim is not None:
                # By schema name: PhysxSchema is Kit's, so the typed
                # PhysxSchema.PhysxSceneAPI.Apply is not importable everywhere,
                # and the applied-schema token is what both sides read.
                prim.AddAppliedSchema("PhysxSceneAPI")
                attributes = {key: (name, type_name) for key, name, type_name in PHYSX_SCENE_SETTINGS}
                for key, value in requested.items():
                    name, type_name = attributes[key]
                    attr = prim.GetAttribute(name)
                    if not attr.IsValid():
                        attr = prim.CreateAttribute(name, getattr(Sdf.ValueTypeNames, type_name), False)
                    attr.Set(value)
                report = _physics_scene_report(stage, prim)
                result = {"prim_path": report.pop("path"), "applied": requested}
                result.update(report)
            print(json.dumps(result))
            """,
        )
        return await self._execute_json_script(script)


def _physx_settings_problem(requested: Dict[str, Any]) -> Optional[str]:
    """Return why a configure_isaac_physics_scene request is invalid, or None.

    Args:
        requested: The settings the caller passed, keyed as in
            ``PHYSX_SCENE_SETTINGS``, token values already upper-cased.

    Returns:
        A message naming the first invalid setting, or None when every value
        is acceptable.
    """
    if not requested:
        return "Pass at least one setting to configure."
    rate = requested.get("time_steps_per_second")
    if rate is not None and rate < 1:
        return f"time_steps_per_second must be >= 1, got {rate!r}"
    if "solver_type" in requested and requested["solver_type"] not in PHYSX_SOLVER_TYPES:
        return f"solver_type must be one of {', '.join(PHYSX_SOLVER_TYPES)}, got {requested['solver_type']!r}"
    if "broadphase_type" in requested and requested["broadphase_type"] not in PHYSX_BROADPHASE_TYPES:
        return (
            f"broadphase_type must be one of {', '.join(PHYSX_BROADPHASE_TYPES)}, "
            f"got {requested['broadphase_type']!r}"
        )
    for kind, lowest in (("position", 1), ("velocity", 0)):
        for bound in ("min", "max"):
            key = f"{bound}_{kind}_iterations"
            value = requested.get(key)
            if value is not None and not lowest <= value <= MAX_SOLVER_ITERATIONS:
                return f"{key} must be in [{lowest}, {MAX_SOLVER_ITERATIONS}], got {value!r}"
        low = requested.get(f"min_{kind}_iterations")
        high = requested.get(f"max_{kind}_iterations")
        if low is not None and high is not None and low > high:
            return f"min_{kind}_iterations ({low}) exceeds max_{kind}_iterations ({high})"
    return None
