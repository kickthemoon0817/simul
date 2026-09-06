"""Compound tools that run several build steps in one Isaac Sim round trip."""

from typing import Any, Dict, List, Optional, Tuple

from ...schemas.common import ErrorResponse
from ._shared import (
    APPLY_COLLISION_CORE,
    APPLY_RIGID_BODY_CORE,
    BIND_MATERIAL_CORE,
    COLLISION_APPROXIMATIONS,
    DEFINE_MATERIAL_CORE,
    DEFINE_PRIM_CORE,
    SET_MASS_PROPERTIES_CORE,
    SET_PRIM_TRANSFORM_CORE,
    FloatList,
    _compose_script,
    _pyval,
)

# Compound-only glue: reuse an existing material or define a fresh
# UsdPreviewSurface one. The single-step tool refuses to overwrite, and a
# caller building an object should be able to point several objects at the
# same look without a separate existence check.
ENSURE_MATERIAL_CORE = """\
    def _ensure_material(stage, material_path, diffuse_color):
        existing = stage.GetPrimAtPath(material_path)
        if existing.IsValid():
            if not existing.IsA(UsdShade.Material):
                return {"error": "Prim at material_path is not a Material: " + material_path}
            return {"material_path": material_path, "created": False}
        return _define_material(stage, material_path, "UsdPreviewSurface", diffuse_color, 0.5, 0.0, 1.0)
"""


class CompoundMixin:
    async def create_isaac_object(
        self,
        prim_path: str,
        prim_type: str = "Cube",
        translation: Optional[FloatList] = None,
        rotation_euler: Optional[FloatList] = None,
        scale: Optional[FloatList] = None,
        rigid_body: bool = False,
        kinematic: bool = False,
        collision: Optional[str] = None,
        mass: Optional[float] = None,
        material_path: Optional[str] = None,
        diffuse_color: Optional[FloatList] = None,
    ) -> Dict[str, Any]:
        """
        Create a prim and configure its transform, physics and look in one call.

        Runs the same script cores as create_isaac_prim,
        set_isaac_prim_transform, add_isaac_rigid_body, add_isaac_collision,
        set_isaac_mass_properties, create_isaac_material and
        assign_isaac_material, in that order, inside a single generated
        script. Steps stop at the first failure; the ones that completed are
        reported so the caller knows what exists on the stage.

        Args:
            prim_path: USD path for the new prim (e.g. "/World/Box").
            prim_type: USD type name (Cube, Sphere, Cylinder, Cone, Capsule,
                Plane, Xform, Mesh, ...).
            translation: Position as [x, y, z].
            rotation_euler: Rotation in degrees as [x, y, z] (XYZ Euler).
            scale: Scale as [x, y, z].
            rigid_body: Apply RigidBodyAPI so the prim is simulated.
            kinematic: With rigid_body, make the body kinematic.
            collision: Collision approximation to apply: "none" for a plain
                CollisionAPI, or one of convexHull, convexDecomposition,
                meshSimplification, boundingSphere, boundingCube. Omit to add
                no collider.
            mass: Mass in kg; applies MassAPI.
            material_path: Material to bind. Reused when it exists, created as
                a UsdPreviewSurface material when it does not. Defaults to
                "<parent>/Looks/<name>_Material" when diffuse_color is given.
            diffuse_color: RGB in 0-1 for a material created by this call.
                Ignored when material_path already exists.

        Returns:
            Dict with ``prim_path`` and a ``steps`` dict keyed by step name
            (prim, transform, rigid_body, collision, mass, material,
            bind_material), plus ``error``/``failed_step`` when a step failed.
        """
        if collision is not None and collision not in COLLISION_APPROXIMATIONS:
            return ErrorResponse(
                error=(
                    f"Invalid collision {collision!r}. Must be one of: "
                    f"{', '.join(COLLISION_APPROXIMATIONS)}"
                ),
                error_type="ValueError",
            ).model_dump()
        if material_path is None and diffuse_color is not None:
            parent, _, name = prim_path.rpartition("/")
            material_path = f"{parent}/Looks/{name}_Material"

        plan: List[Tuple[str, str]] = [
            ("prim", f"_define_prim(stage, prim_path, {_pyval(prim_type)}, {{}})"),
        ]
        if translation is not None or rotation_euler is not None or scale is not None:
            plan.append((
                "transform",
                "_set_prim_transform(stage, prim_path, "
                f"{_pyval(list(translation) if translation else None)}, "
                f"{_pyval(list(rotation_euler) if rotation_euler else None)}, "
                f"{_pyval(list(scale) if scale else None)})",
            ))
        if rigid_body:
            plan.append(("rigid_body", f"_apply_rigid_body(stage, prim_path, {_pyval(bool(kinematic))})"))
        if collision is not None:
            plan.append(("collision", f"_apply_collision(stage, prim_path, {_pyval(collision)})"))
        if mass is not None:
            plan.append(("mass", f"_set_mass_properties(stage, prim_path, {_pyval(float(mass))}, None, None)"))
        if material_path is not None:
            color = [float(c) for c in (diffuse_color or [0.8, 0.8, 0.8])]
            plan.append(("material", f"_ensure_material(stage, material_path, {_pyval(color)})"))
            plan.append(("bind_material", "_bind_material(stage, prim_path, material_path)"))

        # Built at column zero rather than inside a dedented template: the
        # generated list lines would otherwise decide the template's common
        # indent and shift every other line.
        plan_source = "".join(
            f"    ({_pyval(step_name)}, lambda: {call}),\n" for step_name, call in plan
        )
        driver_header = (
            "stage = omni.usd.get_context().get_stage()\n"
            f"prim_path = {_pyval(prim_path)}\n"
            f"material_path = {_pyval(material_path)}\n"
            f"plan = [\n{plan_source}]\n"
        )
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
            """,
            DEFINE_PRIM_CORE,
            SET_PRIM_TRANSFORM_CORE,
            APPLY_RIGID_BODY_CORE,
            APPLY_COLLISION_CORE,
            SET_MASS_PROPERTIES_CORE,
            DEFINE_MATERIAL_CORE,
            BIND_MATERIAL_CORE,
            ENSURE_MATERIAL_CORE,
            driver_header,
            """\
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            else:
                steps = {}
                result = {"prim_path": prim_path, "steps": steps}
                for step_name, run_step in plan:
                    outcome = run_step()
                    if "error" in outcome:
                        result["error"] = outcome["error"]
                        result["failed_step"] = step_name
                        break
                    steps[step_name] = outcome
                print(json.dumps(result))
            """,
        )
        return await self._execute_json_script(script)
