"""Joint drive, joint limit and articulation tools for Isaac Sim.

Everything here is USD-level (UsdPhysics joints, DriveAPI, LimitAPI and
ArticulationRootAPI as authored on the stage), so it reads and writes the
same data whether or not a simulation is running.
"""

import math
from typing import Any, Dict, Optional

from ...schemas.common import ErrorResponse
from ._shared import _compose_script, _pyval
from .._meta import tool_meta

# DriveAPI instance names UsdPhysics defines: "angular" / "linear" for a
# revolute / prismatic joint, the six axes for a generic (D6) joint.
DRIVE_AXES: tuple[str, ...] = (
    "angular", "linear", "transX", "transY", "transZ", "rotX", "rotY", "rotZ",
)
DRIVE_TYPES: tuple[str, ...] = ("force", "acceleration")
# LimitAPI instance names for a generic (D6) joint.
LIMIT_AXES: tuple[str, ...] = (
    "transX", "transY", "transZ", "rotX", "rotY", "rotZ", "distance",
)

# Reads one joint into the payload the joint aspect of get_isaac_prim_detail
# returns. The bridge's PrimDetailReader.joint is the same logic inside Kit;
# tests/isaac/test_prim_detail_bridge.py holds the two to one shape.
# Defines _read_joint(prim).
READ_JOINT_CORE = """\
    def _read_joint(prim):
        import math

        prim_path = str(prim.GetPath())
        if not prim.IsA(UsdPhysics.Joint):
            return {"error": "Prim is not a Joint: " + prim_path}

        def _bound(target, key, value, meaning):
            # UsdPhysics spells "no limit" and "unbreakable" as +-inf, which
            # JSON cannot carry: it is reported as null plus a note.
            if value is None:
                target[key] = None
            elif math.isfinite(float(value)):
                target[key] = float(value)
            else:
                target[key] = None
                target[key + "_note"] = meaning + (" (-inf)" if float(value) < 0 else " (inf)")

        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        enabled = joint.GetJointEnabledAttr().Get()
        result = {
            "prim_path": prim_path,
            "joint_type": prim.GetTypeName(),
            "enabled": enabled if enabled is not None else True,
            "body0": [str(b) for b in body0] if body0 else [],
            "body1": [str(b) for b in body1] if body1 else [],
            "exclude_from_articulation": joint.GetExcludeFromArticulationAttr().Get(),
        }
        _bound(result, "break_force", joint.GetBreakForceAttr().Get(), "unbreakable")
        _bound(result, "break_torque", joint.GetBreakTorqueAttr().Get(), "unbreakable")

        limits = {}
        typed = None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            typed = UsdPhysics.RevoluteJoint(prim)
            limits["type"] = "revolute"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            typed = UsdPhysics.PrismaticJoint(prim)
            limits["type"] = "prismatic"
        if typed is not None:
            limits["axis"] = typed.GetAxisAttr().Get()
            _bound(limits, "lower", typed.GetLowerLimitAttr().Get(), "unlimited")
            _bound(limits, "upper", typed.GetUpperLimitAttr().Get(), "unlimited")
        result["limits"] = limits if limits else None

        limit_axes = {}
        drives = {}
        for schema in prim.GetAppliedSchemas():
            schema_name, _, instance = str(schema).partition(":")
            if schema_name == "PhysicsLimitAPI" and instance:
                limit = UsdPhysics.LimitAPI(prim, instance)
                low = limit.GetLowAttr().Get()
                high = limit.GetHighAttr().Get()
                entry = {"locked": low is not None and high is not None and float(low) > float(high)}
                _bound(entry, "low", low, "unlimited")
                _bound(entry, "high", high, "unlimited")
                limit_axes[instance] = entry
            elif schema_name == "PhysicsDriveAPI" and instance:
                drive = UsdPhysics.DriveAPI(prim, instance)
                drive_type = drive.GetTypeAttr().Get()
                entry = {"type": str(drive_type) if drive_type is not None else "force"}
                _bound(entry, "stiffness", drive.GetStiffnessAttr().Get(), "infinite")
                _bound(entry, "damping", drive.GetDampingAttr().Get(), "infinite")
                _bound(entry, "target_position", drive.GetTargetPositionAttr().Get(), "infinite")
                _bound(entry, "target_velocity", drive.GetTargetVelocityAttr().Get(), "infinite")
                _bound(entry, "max_force", drive.GetMaxForceAttr().Get(), "unlimited")
                drives[instance] = entry
        result["limit_axes"] = limit_axes
        result["drives"] = drives
        return result
"""

# Articulation discovery. Defines _list_articulations(stage, root, limit).
LIST_ARTICULATIONS_CORE = """\
    def _list_articulations(stage, root, limit, offset=0):
        predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

        def _rigid_body_of(path):
            # A body relationship may target a prim below the body; the body
            # is its nearest ancestor with RigidBodyAPI. None is the world.
            if path is None:
                return None
            prim = stage.GetPrimAtPath(path)
            while prim.IsValid() and not prim.IsPseudoRoot():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    return str(prim.GetPath())
                prim = prim.GetParent()
            return None

        def _first_target(rel):
            targets = rel.GetTargets() if rel else []
            return str(targets[0]) if targets else None

        joint_bodies = {}
        joints_by_body = {}
        excluded = set()
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), predicate):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            path = str(prim.GetPath())
            bodies = (
                _rigid_body_of(_first_target(joint.GetBody0Rel())),
                _rigid_body_of(_first_target(joint.GetBody1Rel())),
            )
            joint_bodies[path] = bodies
            if joint.GetExcludeFromArticulationAttr().Get():
                excluded.add(path)
            for body in bodies:
                if body is not None:
                    joints_by_body.setdefault(body, []).append(path)

        def _anchor_joint(body_paths):
            # The joint holding the articulation to the world (one side not
            # a rigid body) makes it fixed-base and comes first.
            for path, (body0, body1) in joint_bodies.items():
                if path in excluded:
                    continue
                if (body0 is None) != (body1 is None) and (body0 or body1) in body_paths:
                    return path
            return None

        def _articulation(root_prim):
            root_path = str(root_prim.GetPath())
            anchor = None
            root_body = None
            if root_prim.IsA(UsdPhysics.Joint):
                body0, body1 = joint_bodies.get(root_path, (None, None))
                root_body = body1 if body1 is not None else body0
                if body0 is None or body1 is None:
                    anchor = root_path
            elif root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                root_body = root_path
                anchor = _anchor_joint({root_path})
            else:
                subtree = [
                    str(p.GetPath()) for p in Usd.PrimRange(root_prim, predicate)
                    if p.HasAPI(UsdPhysics.RigidBodyAPI)
                ]
                anchor = _anchor_joint(set(subtree))
                if anchor is not None:
                    body0, body1 = joint_bodies[anchor]
                    root_body = body0 or body1
                else:
                    children = {
                        joint_bodies[path][1] for path in joint_bodies
                        if path not in excluded and joint_bodies[path][0] in subtree
                    }
                    root_body = next((b for b in subtree if b not in children), subtree[0] if subtree else None)

            ordered = []
            bodies = []
            loop_closures = []
            world_joints = []
            used = set()
            if anchor is not None:
                used.add(anchor)
                ordered.append((anchor, None, root_body))
            if root_body is not None:
                bodies.append(root_body)
            visited = set(bodies)

            def _walk(body):
                # Depth first from the root body, joints in stage order.
                for path in joints_by_body.get(body, []):
                    if path in used or path in excluded:
                        continue
                    used.add(path)
                    body0, body1 = joint_bodies[path]
                    other = body1 if body0 == body else body0
                    if other is None:
                        world_joints.append(path)
                    elif other in visited:
                        loop_closures.append(path)
                    else:
                        visited.add(other)
                        bodies.append(other)
                        ordered.append((path, body, other))
                        _walk(other)

            if root_body is not None:
                _walk(root_body)
            dof_paths, dof_types = _isaac_dofs(root_prim)
            dof_index = {path: index for index, path in enumerate(dof_paths)}
            joints = []
            for index, (path, parent, child) in enumerate(ordered):
                entry = {
                    "tree_index": index,
                    "dof_index": dof_index.get(path),
                    "dof_type": dof_types.get(path),
                    "parent_body": parent,
                    "child_body": child,
                }
                entry.update(_read_joint(stage.GetPrimAtPath(path)))
                joints.append(entry)
            touched = set(bodies)
            excluded_here = sorted(
                path for path in excluded
                if joint_bodies[path][0] in touched or joint_bodies[path][1] in touched
            )
            return {
                "root_path": root_path,
                "root_type": root_prim.GetTypeName(),
                "root_body": root_body,
                "fixed_base": anchor is not None,
                "body_count": len(bodies),
                "bodies": bodies,
                "joint_count": len(joints),
                "joints": joints,
                "dof_count": len(dof_paths),
                "dof_paths": dof_paths,
                "excluded_joints": excluded_here,
                "loop_closure_joints": loop_closures,
                "world_joints": world_joints,
            }

        roots = [
            p for p in Usd.PrimRange(root, predicate) if p.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        page = [_articulation(p) for p in roots[offset:offset + limit]]
        truncated = offset + len(page) < len(roots)
        return {
            "root_path": str(root.GetPath()),
            "articulation_count": len(roots),
            "offset": offset,
            "applied_limit": limit,
            "truncated": truncated,
            "next_offset": offset + len(page) if truncated else None,
            "articulations": page,
        }
"""

# The DOF order Isaac Sim assigns an articulation, as its USD articulation
# backend computes it (Isaac Sim 6.0.1,
# isaacsim.core.experimental.prims/python/impl/_usd_articulation.py:
# _get_root_link_prim, _collect_candidate_joints, _walk_connected_articulation,
# _discover_dofs): breadth first from the root link over the joints under the
# articulation's scope (nested articulations excluded), fixed joints and
# joints without a drivable axis dropped. Defines _isaac_dofs(root_prim) ->
# (dof paths in order, {path: "rotation" | "translation"}).
ISAAC_DOF_ORDER_CORE = """\
    from collections import deque as _dof_deque

    def _isaac_dofs(root_prim):
        stage = root_prim.GetStage()
        root_path = str(root_prim.GetPath())
        scope = root_prim
        if root_prim.IsA(UsdPhysics.Joint) and root_prim.GetParent().IsValid():
            scope = root_prim.GetParent()

        def _target(prim, name):
            rel = prim.GetRelationship(name)
            targets = rel.GetTargets() if rel else []
            if not targets:
                return None
            target = targets[0]
            if not target.IsAbsolutePath():
                target = target.MakeAbsolutePath(prim.GetPath())
            return target.pathString

        def _bodies(joint):
            return _target(joint, "physics:body0"), _target(joint, "physics:body1")

        root_link = None
        if root_prim.IsA(UsdPhysics.Joint):
            for body_path in _bodies(root_prim):
                if body_path and stage.GetPrimAtPath(body_path).IsValid():
                    root_link = stage.GetPrimAtPath(body_path)
                    break
        if root_link is None and root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            root_link = root_prim
        if root_link is None:
            root_link = next((p for p in Usd.PrimRange(scope) if p.HasAPI(UsdPhysics.RigidBodyAPI)), None)
        if root_link is None:
            return [], {}

        def _foreign(prim):
            current = prim
            while current.IsValid():
                if current.HasAPI(UsdPhysics.ArticulationRootAPI) and str(current.GetPath()) != root_path:
                    return True
                if current.GetPath() == scope.GetPath():
                    return False
                current = current.GetParent()
            return False

        by_body = {}
        for prim in Usd.PrimRange(scope):
            if prim.IsA(UsdPhysics.Joint) and not _foreign(prim):
                for slot, body_path in enumerate(_bodies(prim)):
                    if body_path:
                        by_body.setdefault(body_path, []).append((prim, slot))

        joint_paths = []
        seen_links = set()
        seen_joints = set()
        queue = _dof_deque([root_link])
        while queue:
            link = queue.popleft()
            if not link.IsValid():
                continue
            link_path = str(link.GetPath())
            if link_path in seen_links:
                continue
            seen_links.add(link_path)
            for joint, slot in by_body.get(link_path, []):
                joint_path = str(joint.GetPath())
                if joint_path not in seen_joints:
                    seen_joints.add(joint_path)
                    joint_paths.append(joint_path)
                other = _bodies(joint)[1 - slot]
                if other and other not in seen_links and stage.GetPrimAtPath(other).IsValid():
                    queue.append(stage.GetPrimAtPath(other))
        if root_path not in seen_joints and root_prim.IsA(UsdPhysics.Joint):
            joint_paths.append(root_path)

        dof_paths = []
        dof_types = {}
        for joint_path in joint_paths:
            joint = stage.GetPrimAtPath(joint_path)
            if not joint.IsValid() or joint.IsA(UsdPhysics.FixedJoint):
                continue
            kind = None
            if joint.IsA(UsdPhysics.RevoluteJoint) or joint.IsA(UsdPhysics.SphericalJoint):
                kind = "rotation"
            elif joint.IsA(UsdPhysics.PrismaticJoint):
                kind = "translation"
            else:
                # A generic joint counts by its first drive attribute only.
                for attr in joint.GetAttributes():
                    name = attr.GetName()
                    if not name.startswith("drive:"):
                        continue
                    if name.startswith("drive:linear:"):
                        kind = "translation"
                    elif name.startswith("drive:angular:"):
                        kind = "rotation"
                    break
            if kind is not None:
                dof_paths.append(joint_path)
                dof_types[joint_path] = kind
        return dof_paths, dof_types
"""

# Joint writers. Define _set_joint_drive(...) and _set_joint_limits(...);
# both read the joint back through _read_joint.
SET_JOINT_CORE = """\
    def _joint_kind(prim):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            return "revolute"
        if prim.IsA(UsdPhysics.PrismaticJoint):
            return "prismatic"
        if prim.IsA(UsdPhysics.SphericalJoint):
            return "spherical"
        if prim.IsA(UsdPhysics.FixedJoint):
            return "fixed"
        if prim.IsA(UsdPhysics.DistanceJoint):
            return "distance"
        return "generic"

    def _editable_joint(stage, joint_path):
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            return None, "Prim not found: " + joint_path
        if not prim.IsA(UsdPhysics.Joint):
            return None, "Prim is not a Joint: " + joint_path
        if prim.IsInstanceProxy():
            return None, "Joint is inside an instance and cannot be edited here: " + joint_path
        return prim, None

    DRIVE_AXES_BY_KIND = {
        "revolute": ("angular",),
        "prismatic": ("linear",),
        "spherical": ("rotX", "rotY", "rotZ"),
        "generic": ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"),
        "fixed": (),
        "distance": (),
    }

    def _set_joint_drive(stage, joint_path, axis, values):
        prim, error = _editable_joint(stage, joint_path)
        if error is not None:
            return {"error": error}
        kind = _joint_kind(prim)
        allowed = DRIVE_AXES_BY_KIND[kind]
        if not allowed:
            return {"error": "A " + kind + " joint has no degree of freedom to drive: " + joint_path}
        if axis is None:
            if len(allowed) != 1:
                return {"error": (
                    "A " + kind + " joint needs drive_axis, one of: " + ", ".join(allowed)
                )}
            axis = allowed[0]
        if axis not in allowed:
            return {"error": (
                "drive_axis " + axis + " does not apply to a " + kind + " joint; use one of: "
                + ", ".join(allowed)
            )}
        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        writers = {
            "type": drive.CreateTypeAttr,
            "stiffness": drive.CreateStiffnessAttr,
            "damping": drive.CreateDampingAttr,
            "target_position": drive.CreateTargetPositionAttr,
            "target_velocity": drive.CreateTargetVelocityAttr,
            "max_force": drive.CreateMaxForceAttr,
        }
        for key, value in values.items():
            writers[key]().Set(value if key == "type" else float(value))
        result = _read_joint(prim)
        result["drive_axis"] = axis
        result["applied"] = values
        return result

    def _set_joint_limits(stage, joint_path, axis, lower, upper):
        prim, error = _editable_joint(stage, joint_path)
        if error is not None:
            return {"error": error}
        kind = _joint_kind(prim)
        if kind in ("revolute", "prismatic"):
            if axis is not None:
                return {"error": (
                    "A " + kind + " joint has one limited axis; omit limit_axis and pass lower/upper"
                )}
            typed = UsdPhysics.RevoluteJoint(prim) if kind == "revolute" else UsdPhysics.PrismaticJoint(prim)
            low_attr, high_attr = typed.GetLowerLimitAttr(), typed.GetUpperLimitAttr()
            create_low, create_high = typed.CreateLowerLimitAttr, typed.CreateUpperLimitAttr
        elif kind == "generic":
            if axis is None:
                return {"error": (
                    "A generic (D6) joint needs limit_axis, one of: "
                    "transX, transY, transZ, rotX, rotY, rotZ, distance"
                )}
            limit = UsdPhysics.LimitAPI.Apply(prim, axis)
            low_attr, high_attr = limit.GetLowAttr(), limit.GetHighAttr()
            create_low, create_high = limit.CreateLowAttr, limit.CreateHighAttr
        else:
            return {"error": (
                "Limits of a " + kind + " joint are not set by this tool: " + joint_path
                + " (spherical joints take cone angles; fixed joints have no free axis)"
            )}
        new_low = lower if lower is not None else low_attr.Get()
        new_high = upper if upper is not None else high_attr.Get()
        if (
            kind != "generic"
            and new_low is not None and new_high is not None
            and float(new_low) > float(new_high)
        ):
            return {"error": (
                "lower (" + str(new_low) + ") would exceed upper (" + str(new_high) + ") on "
                + joint_path + "; nothing was changed"
            )}
        if lower is not None:
            create_low().Set(float(lower))
        if upper is not None:
            create_high().Set(float(upper))
        result = _read_joint(prim)
        result["limit_axis"] = axis
        result["applied"] = {key: value for key, value in (("lower", lower), ("upper", upper)) if value is not None}
        return result
"""


def _finite_problem(name: str, value: Optional[float], minimum: Optional[float] = None) -> Optional[str]:
    """Return why ``value`` is not an acceptable number, or None.

    Args:
        name: Parameter name for the message.
        value: The value passed, or None when it was omitted.
        minimum: Smallest allowed value, or None for no lower bound.

    Returns:
        An error message, or None when the value is acceptable or omitted.
    """
    if value is None:
        return None
    if not math.isfinite(value):
        return f"{name} must be a finite number, got {value!r}"
    if minimum is not None and value < minimum:
        return f"{name} must be >= {minimum}, got {value!r}"
    return None


class JointsMixin:
    # ------------------------------------------------------------------
    # Joints and articulations (USD-level)
    # ------------------------------------------------------------------

    @tool_meta(
        name="list_isaac_articulations",
        description=(
            "Find articulations (prims with ArticulationRootAPI) under root_path and list each "
            "one's joints in kinematic order, depth first from the root body (tree_index), "
            "with dof_index, the position of the joint's DOF in Isaac Sim's articulation DOF "
            "order (breadth first from the root link, fixed joints dropped; null for joints "
            "without a DOF), joint type, body0/body1, parent and child body, axis, limits and "
            "drives (stiffness, damping, targets, max force). Each articulation also lists "
            "dof_paths in DOF order, the bodies, whether the base is fixed to the world, and "
            "joints excluded from it or closing a loop. Paged by max_results / offset."
        ),
        read_only=True,
        idempotent=True,
    )
    async def list_isaac_articulations(
        self, root_path: str = "/", max_results: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """
        List articulations and their joints in kinematic order.

        Args:
            root_path: USD path to search under for ArticulationRootAPI
                prims. Joints and bodies are followed wherever they are on
                the stage.
            max_results: Maximum number of articulations returned per page,
                clamped to [1, 200]; ``articulation_count`` is the full
                number.
            offset: Number of articulations to skip before the page starts;
                pass the previous page's ``next_offset`` to continue.

        Returns:
            Dict with one entry per articulation: ``root_body``,
            ``fixed_base``, ``bodies``, ``dof_paths`` (Isaac Sim's DOF
            order) and ``joints`` in tree order (each joint with
            ``tree_index``, ``dof_index``, ``dof_type``, ``parent_body``,
            ``child_body`` and the joint aspect's fields), plus
            ``excluded_joints``, ``loop_closure_joints`` and
            ``world_joints``; ``next_offset`` when more articulations remain.
        """
        max_results = max(1, min(max_results, 200))
        offset = max(0, offset)
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics
            """,
            READ_JOINT_CORE,
            ISAAC_DOF_ORDER_CORE,
            LIST_ARTICULATIONS_CORE,
            f"""\
            root_path = {_pyval(root_path)}
            limit = {max_results}
            offset = {offset}
            """,
            """\
            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath(root_path) if stage is not None else None
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            elif not root.IsValid():
                print(json.dumps({"error": "Root path not found: " + root_path}))
            else:
                print(json.dumps(_list_articulations(stage, root, limit, offset)))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="set_isaac_joint_drive",
        description=(
            "Set a joint drive (UsdPhysics.DriveAPI): stiffness, damping, target_position, "
            "target_velocity, max_force and drive_type (force or acceleration). drive_axis "
            "defaults to angular for a revolute joint and linear for a prismatic one; a "
            "generic (D6) or spherical joint needs it (transX..rotZ). Angular targets are in "
            "degrees and degrees/s, linear ones in stage units. Applies DriveAPI when missing, "
            "writes only the values passed and returns the joint's drives and limits."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def set_isaac_joint_drive(
        self,
        joint_path: str,
        drive_axis: Optional[str] = None,
        stiffness: Optional[float] = None,
        damping: Optional[float] = None,
        target_position: Optional[float] = None,
        target_velocity: Optional[float] = None,
        max_force: Optional[float] = None,
        drive_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Author a joint's drive.

        Args:
            joint_path: USD path of the joint prim.
            drive_axis: DriveAPI instance: "angular" (revolute), "linear"
                (prismatic), or transX/transY/transZ/rotX/rotY/rotZ for a
                generic (D6) or spherical joint. Defaults to the only
                choice for revolute and prismatic joints.
            stiffness: Spring stiffness (>= 0) toward target_position.
            damping: Damping (>= 0) toward target_velocity.
            target_position: Position target, degrees for angular drives
                and stage units for linear ones.
            target_velocity: Velocity target, degrees/s or stage units/s.
            max_force: Maximum force or torque the drive applies (>= 0).
            drive_type: "force" (stiffness and damping are forces) or
                "acceleration" (they are scaled by the body's inertia).

        Returns:
            The joint as the joint aspect reports it (drives, limits,
            bodies), plus ``drive_axis`` and ``applied``.
        """
        values: Dict[str, Any] = {
            key: value
            for key, value in (
                ("type", drive_type),
                ("stiffness", stiffness),
                ("damping", damping),
                ("target_position", target_position),
                ("target_velocity", target_velocity),
                ("max_force", max_force),
            )
            if value is not None
        }
        problem: Optional[str] = None
        if not values:
            problem = "Pass at least one drive value to set."
        elif drive_axis is not None and drive_axis not in DRIVE_AXES:
            problem = f"drive_axis must be one of {', '.join(DRIVE_AXES)}, got {drive_axis!r}"
        elif drive_type is not None and drive_type not in DRIVE_TYPES:
            problem = f"drive_type must be one of {', '.join(DRIVE_TYPES)}, got {drive_type!r}"
        else:
            for name, value, minimum in (
                ("stiffness", stiffness, 0.0),
                ("damping", damping, 0.0),
                ("max_force", max_force, 0.0),
                ("target_position", target_position, None),
                ("target_velocity", target_velocity, None),
            ):
                problem = _finite_problem(name, value, minimum)
                if problem is not None:
                    break
        if problem is not None:
            return ErrorResponse(error=problem, error_type="ValidationError").model_dump()
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics
            """,
            READ_JOINT_CORE,
            SET_JOINT_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_set_joint_drive(
                    stage, {_pyval(joint_path)}, {_pyval(drive_axis)}, {_pyval(values)}
                )))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="set_isaac_joint_limits",
        description=(
            "Set a joint's limits. Revolute (degrees) and prismatic (stage units) joints take "
            "lower and upper directly; a generic (D6) joint takes them per limit_axis "
            "(transX..rotZ, distance) through UsdPhysics.LimitAPI, where lower > upper locks "
            "the axis. Writes only the bounds passed, refuses lower > upper on revolute and "
            "prismatic joints, and returns the joint's limits and drives."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def set_isaac_joint_limits(
        self,
        joint_path: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        limit_axis: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Author a joint's lower and upper limits.

        Args:
            joint_path: USD path of the joint prim.
            lower: Lower limit: degrees for revolute joints and rotational
                axes, stage units otherwise. Omit to keep the current one.
            upper: Upper limit, same units. Omit to keep the current one.
            limit_axis: For a generic (D6) joint, the LimitAPI axis:
                transX, transY, transZ, rotX, rotY, rotZ or distance. Omit
                for revolute and prismatic joints.

        Returns:
            The joint as the joint aspect reports it, plus ``limit_axis``
            and ``applied``.
        """
        problem: Optional[str] = None
        if lower is None and upper is None:
            problem = "Pass lower, upper or both."
        elif limit_axis is not None and limit_axis not in LIMIT_AXES:
            problem = f"limit_axis must be one of {', '.join(LIMIT_AXES)}, got {limit_axis!r}"
        else:
            problem = _finite_problem("lower", lower) or _finite_problem("upper", upper)
        if problem is not None:
            return ErrorResponse(error=problem, error_type="ValidationError").model_dump()
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics
            """,
            READ_JOINT_CORE,
            SET_JOINT_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_set_joint_limits(
                    stage, {_pyval(joint_path)}, {_pyval(limit_axis)}, {_pyval(lower)}, {_pyval(upper)}
                )))
            """,
        )
        return await self._execute_json_script(script)
