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
        self, root_path: str = "/", max_results: int = DEFAULT_MAX_RESULTS
    ) -> Dict[str, Any]:
        """
        List all prims with physics APIs applied in the stage.

        Args:
            root_path: Root path to search under.
            max_results: Maximum entries returned per list (rigid bodies,
                colliders, joints). The counts always cover the whole subtree.

        Returns:
            Dict with lists of rigid bodies, colliders, and joints.
        """
        max_results = max(1, min(max_results, 10000))
        _root_path = _pyval(root_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                root = stage.GetPrimAtPath({_root_path})
                if not root.IsValid():
                    print(json.dumps({{"error": "Root path not found: " + {_root_path}}}))
                else:
                    max_results = {max_results}
                    counts = {{"rigid_bodies": 0, "colliders": 0, "joints": 0}}
                    found = {{"rigid_bodies": [], "colliders": [], "joints": []}}
                    for p in Usd.PrimRange(root):
                        kinds = []
                        if p.HasAPI(UsdPhysics.RigidBodyAPI):
                            kinds.append("rigid_bodies")
                        if p.HasAPI(UsdPhysics.CollisionAPI):
                            kinds.append("colliders")
                        if p.IsA(UsdPhysics.Joint):
                            kinds.append("joints")
                        if not kinds:
                            continue
                        entry = {{"path": str(p.GetPath()), "type": p.GetTypeName()}}
                        for kind in kinds:
                            counts[kind] += 1
                            if len(found[kind]) < max_results:
                                found[kind].append(entry)
                    print(json.dumps({{
                        "root_path": {_root_path},
                        "rigid_body_count": counts["rigid_bodies"],
                        "collider_count": counts["colliders"],
                        "joint_count": counts["joints"],
                        "max_results": max_results,
                        "truncated": any(counts[k] > max_results for k in counts),
                        "rigid_bodies": found["rigid_bodies"],
                        "colliders": found["colliders"],
                        "joints": found["joints"],
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
            mass: Mass value in kg.
            density: Density value.
            center_of_mass: Center of mass as [x, y, z].

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
        _prim_path = _pyval(prim_path)
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

