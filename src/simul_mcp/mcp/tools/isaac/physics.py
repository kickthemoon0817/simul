"""Physics Inspection tools for Isaac Sim."""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ....adapters import IsaacSocketClient, ScriptResult
from ...schemas.common import ErrorResponse
from ._shared import (
    APPLY_COLLISION_CORE,
    APPLY_RIGID_BODY_CORE,
    BULK_GEOMETRY_ATTRIBUTES,
    DEFAULT_MAX_RESULTS,
    LOG_SCAN_WINDOW_BYTES,
    MAX_CAPTURE_DIMENSION,
    MAX_INLINE_CAPTURE_BYTES,
    MAX_RETAINED_CAPTURES,
    MAX_SCRIPT_BYTES,
    PRIM_DETAIL_ASPECTS,
    SET_MASS_PROPERTIES_CORE,
    FloatList,
    _compose_script,
    _pyval,
    logger,
)


class PhysicsMixin:
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
        _prim_path = _pyval(prim_path)
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
            Dict with joint type, bodies, limits, and drive settings.
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
        _prim_path = _pyval(prim_path)
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
        _prim_path = _pyval(prim_path)
        _approx = _pyval(approximation)
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import UsdPhysics
            """,
            APPLY_COLLISION_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_apply_collision(stage, {_prim_path}, {_approx})))
            """,
        )
        return await self._execute_json_script(script)

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
        _prim_path = _pyval(prim_path)
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
        gravity_direction: Optional[FloatList] = None,
        gravity_magnitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Create a UsdPhysics.Scene prim, or update the gravity of an existing one.

        Args:
            prim_path: USD path for the physics scene prim.
            gravity_direction: Gravity direction as a unit vector [x, y, z] in
                stage axes; Isaac Sim stages are Z-up, so straight down is
                [0, 0, -1]. That is the default for a new scene; an existing
                scene keeps its direction unless one is passed.
            gravity_magnitude: Gravity magnitude in m/s^2 (positive, along
                gravity_direction). Defaults to 9.81 for a new scene; an
                existing scene keeps its magnitude unless one is passed.

        Returns:
            Dict with the scene's gravity, ``created`` for a new prim, and
            ``already_existed`` plus ``updated`` for a scene that was there.
        """
        _prim_path = _pyval(prim_path)
        _requested_direction = _pyval(
            [float(component) for component in gravity_direction]
            if gravity_direction is not None
            else None
        )
        _requested_magnitude = _pyval(
            float(gravity_magnitude) if gravity_magnitude is not None else None
        )
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdPhysics, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                requested_direction = {_requested_direction}
                requested_magnitude = {_requested_magnitude}

                def _gravity_of(scene):
                    direction = scene.GetGravityDirectionAttr().Get()
                    magnitude = scene.GetGravityMagnitudeAttr().Get()
                    return {{
                        "gravity_direction": (
                            [float(c) for c in direction] if direction is not None else None
                        ),
                        "gravity_magnitude": (
                            float(magnitude) if magnitude is not None else None
                        ),
                    }}

                existing = stage.GetPrimAtPath({_prim_path})
                if existing.IsValid() and existing.IsA(UsdPhysics.Scene):
                    scene = UsdPhysics.Scene(existing)
                    updated = False
                    if requested_direction is not None:
                        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(*requested_direction))
                        updated = True
                    if requested_magnitude is not None:
                        scene.CreateGravityMagnitudeAttr().Set(requested_magnitude)
                        updated = True
                    result = {{
                        "prim_path": {_prim_path},
                        "already_existed": True,
                        "created": False,
                        "updated": updated,
                        "message": (
                            "Physics scene already exists at this path; gravity updated"
                            if updated
                            else "Physics scene already exists at this path; gravity unchanged"
                        ),
                    }}
                    result.update(_gravity_of(scene))
                    print(json.dumps(result))
                else:
                    scene_prim = stage.DefinePrim({_prim_path}, "PhysicsScene")
                    if not scene_prim.IsValid():
                        print(json.dumps({{"error": "Failed to create physics scene prim"}}))
                    else:
                        scene = UsdPhysics.Scene(scene_prim)
                        direction = requested_direction or [0.0, 0.0, -1.0]
                        magnitude = requested_magnitude if requested_magnitude is not None else 9.81
                        scene.CreateGravityDirectionAttr(Gf.Vec3f(*direction))
                        scene.CreateGravityMagnitudeAttr(magnitude)
                        result = {{
                            "prim_path": {_prim_path},
                            "already_existed": False,
                            "created": True,
                            "updated": False,
                        }}
                        result.update(_gravity_of(scene))
                        print(json.dumps(result))
        """)
        return await self._execute_json_script(script)

