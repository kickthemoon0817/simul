"""Unit correction for the physics values of a referenced asset.

Scaling the prim that references a centimetre asset into a metre stage fixes
its geometry, but PhysX reads physics values (prismatic limits, drive
targets and gains, density, inertia, velocities, contact offsets) in stage
units whatever the transform says. Kit's own metrics assembler
(``omni.usd.metrics.assembler.physics``) rescales exactly those values, with
the exponents in ``pxr.PhysicsSchemaTools.units_db``; it runs only for
references made through Kit's reference commands, never for the raw pxr
``AddReference`` simul uses, so simul does the same rescaling itself.

``PHYSICS_UNITS_CORE`` is the Kit-side script. It prefers Kit's own table
(``pxr.PhysicsSchemaTools.units.get_attribute_unit_exponents``) and falls
back to ``PHYSICS_UNIT_EXPONENTS``, a subset of the same table covering the
UsdPhysics schemas and the PhysX attributes robot assets carry. Each entry is
``(mass exponent, distance exponent)``, the two indices of the units_db
tuples that depend on kilogramsPerUnit and metersPerUnit; angles and time do
not change between stages.
"""

import textwrap
from typing import Dict, Tuple, Union

from ._shared import _pyval

ExponentSpec = Union[Tuple[int, int], Dict[str, Tuple[int, int]]]

# (mass, distance) exponents per attribute, from units_db.usd_attribute_units.
# A dict entry has variants resolved per prim, as units.get_attribute_unit_exponents does.
PHYSICS_UNIT_EXPONENTS: Dict[str, ExponentSpec] = {
    # UsdPhysics
    "physics:gravityMagnitude": (0, 1),
    "physics:velocity": (0, 1),
    "physics:mass": (1, 0),
    "physics:density": (1, -3),
    "physics:diagonalInertia": (1, 2),
    "physics:breakForce": (1, 1),
    "physics:breakTorque": (1, 2),
    "physics:lowerLimit": {"RevoluteJoint": (0, 0), "PrismaticJoint": (0, 1)},
    "physics:upperLimit": {"RevoluteJoint": (0, 0), "PrismaticJoint": (0, 1)},
    "physics:minDistance": (0, 1),
    "physics:maxDistance": (0, 1),
    # PhysX scene, bodies, collision
    "physxScene:bounceThreshold": (0, 1),
    "physxScene:frictionOffsetThreshold": (0, 1),
    "physxScene:frictionCorrelationDistance": (0, 1),
    "physxRigidBody:maxLinearVelocity": (0, 1),
    "physxRigidBody:sleepThreshold": (0, 2),
    "physxRigidBody:stabilizationThreshold": (0, 2),
    "physxRigidBody:maxDepenetrationVelocity": (0, 1),
    "physxRigidBody:maxContactImpulse": (1, 1),
    "physxRigidBody:contactSlopCoefficient": (0, 1),
    "physxContactReport:threshold": (1, 1),
    "physxCollision:contactOffset": (0, 1),
    "physxCollision:restOffset": (0, 1),
    "physxCollision:torsionalPatchRadius": (0, 1),
    "physxCollision:minTorsionalPatchRadius": (0, 1),
    "physxConvexHullCollision:minThickness": (0, 1),
    "physxConvexDecompositionCollision:minThickness": (0, 1),
    "physxTriangleMeshSimplificationCollision:weldTolerance": (0, 1),
    "physxTriangleMeshCollision:weldTolerance": (0, 1),
    "physxSurfaceVelocity:surfaceVelocity": (0, 1),
    "physxMaterial:compliantContactStiffness": {"Force": (1, 0), "Acceleration": (0, 0)},
    "physxMaterial:compliantContactDamping": {"Force": (1, 0), "Acceleration": (0, 0)},
    # PhysX joints and articulations
    "physxJoint:maxJointVelocity": {"Linear joint": (0, 1), "Angular joint": (0, 0)},
    "physxJoint:armature": {"Linear joint": (1, 0), "Angular joint": (1, 2)},
    "physxPhysicsDistanceJoint:springStiffness": (1, 0),
    "physxPhysicsDistanceJoint:springDamping": (1, 0),
    "physxArticulation:sleepThreshold": (0, 2),
    "physxArticulation:stabilizationThreshold": (0, 2),
    "physxForce:force": {"Force": (1, 1), "Acceleration": (0, 1)},
    "physxForce:torque": {"Force": (1, 2), "Acceleration": (0, 0)},
}

# Multiple-apply schemas, per instance: limits, drives, joint state, PhysX
# limits and mimic joints. Linear instances carry distance, rotational ones
# angles (which do not scale); the drive gains depend on the drive type.
_LINEAR_INSTANCES = ("transX", "transY", "transZ", "distance", "linear")
_ANGULAR_INSTANCES = ("rotX", "rotY", "rotZ", "angular")
for _inst in _LINEAR_INSTANCES:
    PHYSICS_UNIT_EXPONENTS["state:" + _inst + ":physics:position"] = (0, 1)
    PHYSICS_UNIT_EXPONENTS["state:" + _inst + ":physics:velocity"] = (0, 1)
    PHYSICS_UNIT_EXPONENTS["physxLimit:" + _inst + ":bounceThreshold"] = (0, 1)
    PHYSICS_UNIT_EXPONENTS["physxLimit:" + _inst + ":stiffness"] = (1, 0)
    PHYSICS_UNIT_EXPONENTS["physxLimit:" + _inst + ":damping"] = (1, 0)
    if _inst != "linear":
        PHYSICS_UNIT_EXPONENTS["limit:" + _inst + ":physics:low"] = (0, 1)
        PHYSICS_UNIT_EXPONENTS["limit:" + _inst + ":physics:high"] = (0, 1)
    if _inst != "distance":
        PHYSICS_UNIT_EXPONENTS["drive:" + _inst + ":physics:maxForce"] = (1, 1)
        PHYSICS_UNIT_EXPONENTS["drive:" + _inst + ":physics:targetPosition"] = (0, 1)
        PHYSICS_UNIT_EXPONENTS["drive:" + _inst + ":physics:targetVelocity"] = (0, 1)
        PHYSICS_UNIT_EXPONENTS["drive:" + _inst + ":physics:stiffness"] = {"Force": (1, 0), "Acceleration": (0, 0)}
        PHYSICS_UNIT_EXPONENTS["drive:" + _inst + ":physics:damping"] = {"Force": (1, 0), "Acceleration": (0, 0)}
for _inst in _ANGULAR_INSTANCES:
    PHYSICS_UNIT_EXPONENTS["physxLimit:" + _inst + ":stiffness"] = (1, 2)
    PHYSICS_UNIT_EXPONENTS["physxLimit:" + _inst + ":damping"] = (1, 2)
    PHYSICS_UNIT_EXPONENTS["drive:" + _inst + ":physics:maxForce"] = (1, 2)
    PHYSICS_UNIT_EXPONENTS["drive:" + _inst + ":physics:stiffness"] = {"Force": (1, 2), "Acceleration": (0, 2)}
    PHYSICS_UNIT_EXPONENTS["drive:" + _inst + ":physics:damping"] = {"Force": (1, 2), "Acceleration": (0, 2)}
for _inst in ("rotX", "rotY", "rotZ"):
    PHYSICS_UNIT_EXPONENTS["physxMimicJoint:" + _inst + ":gearing"] = (0, 1)
del _inst

# PhysX tendon attributes, physxTendon:<instance>:<name>. units_db lists them
# by bare name, which a full-name lookup does not find, so they are matched
# on the last component here whichever table is in use.
PHYSX_TENDON_UNIT_EXPONENTS: Dict[str, Tuple[int, int]] = {
    "stiffness": (1, 0),
    "damping": (1, 0),
    "limitStiffness": (1, 0),
    "offset": (0, 1),
    "restLength": (0, 1),
    "lowerLimit": (0, 1),
    "upperLimit": (0, 1),
    "localPos": (0, 1),
}

# Reads (and, when asked, rescales) the physics values under a prim. Defines
# _physics_unit_correction(stage, root, length_ratio, mass_ratio, apply).
# Needs PHYSICS_UNIT_EXPONENTS from physics_units_source().
PHYSICS_UNITS_CORE = """\
    import math as _units_math

    try:
        from pxr.PhysicsSchemaTools.units import get_attribute_unit_exponents as _kit_unit_exponents
    except Exception:
        _kit_unit_exponents = None

    def _fallback_unit_exponents(name, prim):
        spec = PHYSICS_UNIT_EXPONENTS.get(name)
        if not isinstance(spec, dict):
            return spec
        if "RevoluteJoint" in spec:
            return spec["RevoluteJoint"] if prim.IsA(UsdPhysics.RevoluteJoint) else spec["PrismaticJoint"]
        if "Angular joint" in spec:
            angular = prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.SphericalJoint)
            return spec["Angular joint"] if angular else spec["Linear joint"]
        if name.startswith("drive:"):
            kind = prim.GetAttribute("drive:" + name.split(":")[1] + ":physics:type")
            mode = kind.Get() if kind.IsValid() else None
            return spec["Force"] if mode == "force" else spec["Acceleration"]
        if name.startswith("physxMaterial:"):
            spring = prim.GetAttribute("physxMaterial:compliantContactAccelerationSpring")
            return spec["Acceleration"] if spring.IsValid() and spring.Get() is True else spec["Force"]
        if name.startswith("physxForce:"):
            mode = prim.GetAttribute("physxForce:mode")
            return spec["Force"] if mode.IsValid() and mode.Get() == "force" else spec["Acceleration"]
        return None

    def _unit_exponents(name, prim):
        # (mass, distance) for an attribute, from Kit's table when it is
        # there (units_db tuples are mass, distance, degrees, seconds,
        # radians), else from the built-in subset of it.
        parts = name.split(":")
        if len(parts) == 3 and parts[0] == "physxTendon":
            return PHYSX_TENDON_UNIT_EXPONENTS.get(parts[2])
        if _kit_unit_exponents is not None:
            try:
                exponents = _kit_unit_exponents(name, prim)
            except Exception:
                exponents = None
            if exponents is not None and not isinstance(exponents, dict):
                return int(exponents[0]), int(exponents[1])
            if exponents is not None:
                return None
        exponents = _fallback_unit_exponents(name, prim)
        return (int(exponents[0]), int(exponents[1])) if exponents is not None else None

    def _scaled(value, factor):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value) * factor
        if hasattr(value, "__len__") and not isinstance(value, str):
            try:
                return type(value)([_scaled(item, factor) for item in value])
            except Exception:
                return type(value)(*[float(item) * factor for item in value])
        return None

    def _physics_unit_correction(stage, root, length_ratio, mass_ratio, apply, limit=50):
        scaled = []
        uncorrected = []
        by_attribute = {}
        # Instance proxies are visited to report what cannot be edited here.
        predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        for prim in Usd.PrimRange(root, predicate):
            for attr in prim.GetAttributes():
                name = attr.GetName()
                exponents = _unit_exponents(name, prim)
                if exponents is None or exponents == (0, 0) or not attr.HasAuthoredValue():
                    continue
                factor = (length_ratio ** exponents[1]) * (mass_ratio ** exponents[0])
                if _units_math.isclose(factor, 1.0, rel_tol=1e-12):
                    continue
                where = {"path": str(prim.GetPath()), "attribute": name}
                if prim.IsInstanceProxy():
                    uncorrected.append(dict(where, reason="inside an instance; correct it in the source asset"))
                    continue
                if attr.GetNumTimeSamples() > 0:
                    uncorrected.append(dict(where, reason="time-sampled; only default values are corrected"))
                    continue
                value = attr.Get()
                new_value = _scaled(value, factor)
                if new_value is None:
                    uncorrected.append(dict(where, reason="value of type " + str(attr.GetTypeName()) + " not scaled"))
                    continue
                if apply and not attr.Set(new_value):
                    uncorrected.append(dict(where, reason="USD refused the corrected value"))
                    continue
                entry = by_attribute.setdefault(name, {"count": 0, "factor": factor})
                entry["count"] += 1
                if len(scaled) < limit:
                    def _plain(v):
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            return float(v) if _units_math.isfinite(float(v)) else None
                        if hasattr(v, "__len__") and not isinstance(v, str):
                            return [_plain(item) for item in v][:8]
                        return str(v)
                    scaled.append(dict(where, before=_plain(value), after=_plain(new_value)))
        return {
            "applied": bool(apply),
            "table": "pxr.PhysicsSchemaTools.units_db" if _kit_unit_exponents is not None else "simul built-in subset",
            "length_ratio": length_ratio,
            "mass_ratio": mass_ratio,
            "attribute_count": sum(entry["count"] for entry in by_attribute.values()),
            "by_attribute": by_attribute,
            "examples": scaled,
            "uncorrected": uncorrected[:limit],
            "uncorrected_count": len(uncorrected),
        }
"""


def physics_units_source() -> str:
    """Return ``PHYSICS_UNITS_CORE`` with the table it reads, as script source.

    Returns:
        Python source defining ``_physics_unit_correction`` for a generated
        script that imports ``Usd`` and ``UsdPhysics``.
    """
    return (
        f"PHYSICS_UNIT_EXPONENTS = {_pyval(PHYSICS_UNIT_EXPONENTS)}\n"
        f"PHYSX_TENDON_UNIT_EXPONENTS = {_pyval(PHYSX_TENDON_UNIT_EXPONENTS)}\n"
        + textwrap.dedent(PHYSICS_UNITS_CORE)
    )
