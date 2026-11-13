"""Compound tools that run several build steps in one Isaac Sim round trip."""

from typing import Any, Dict, List, Optional, Tuple

from ...schemas.common import ErrorResponse
from ._shared import (
    APPLY_COLLISION_CORE,
    APPLY_RIGID_BODY_CORE,
    BIND_MATERIAL_CORE,
    COERCE_ATTRIBUTE_VALUE_CORE,
    COLLISION_APPROXIMATIONS,
    DEFINE_MATERIAL_CORE,
    DEFINE_PRIM_CORE,
    SET_MASS_PROPERTIES_CORE,
    SET_PRIM_TRANSFORM_CORE,
    XFORM_READ_CORE,
    FloatList,
    _compose_script,
    _pyval,
)
from .._meta import tool_meta
from .graph import _script_execution_refusal, is_script_capable_node_type

# Largest script create_isaac_object sends in one round trip. Isaac Sim 5.1's
# stock socket takes at most 32 KiB per script (less an auth-token header);
# every helper on every call reached 31.8-32.4 KB and went over with long
# paths. Steps that don't fit start another script.
OBJECT_SCRIPT_BUDGET = 20_000

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
    @tool_meta(
        name="create_isaac_object",
        description=(
            "Build a complete object in one call: create a prim, then optionally set its "
            "transform, apply a rigid body, a collider, a mass, and bind a material (reused "
            "when material_path exists, otherwise created with diffuse_color). Prefer this "
            "over chaining create_isaac_prim, set_isaac_prim_transform, "
            "add_isaac_rigid_body, add_isaac_collision, set_isaac_mass_properties, "
            "create_isaac_material and assign_isaac_material: it runs the same steps in one "
            "round trip (two for the full set, to stay inside Isaac Sim 5.1's script size "
            "limit; reported as round_trips) and reports each step's result under 'steps'. "
            "Stops at the "
            "first failing step and names it in 'failed_step'; what the earlier steps "
            "created stays on the stage and is listed in 'left_on_stage'. Vector "
            "arguments must have 3 components (checked before anything is created)."
        ),
        read_only=False,
    )
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
            collision: Add a collider, with this approximation for a Mesh:
                "none" (the triangle mesh), convexHull, convexDecomposition,
                meshSimplification, boundingSphere or boundingCube. Implicit
                shapes (Cube, Sphere, ...) collide with their exact shape
                whatever the value; an Xform has no geometry and fails the
                step. Omit to add no collider.
            mass: Mass in kg; applies MassAPI.
            material_path: Material to bind. Reused when it exists, created as
                a UsdPreviewSurface material when it does not. Defaults to
                "<parent>/Looks/<name>_Material" when diffuse_color is given.
            diffuse_color: RGB in 0-1 for a material created by this call.
                Ignored when material_path already exists.

        Returns:
            Dict with ``prim_path`` and a ``steps`` dict keyed by step name
            (prim, transform, rigid_body, collision, mass, material,
            bind_material), plus ``error``/``failed_step`` when a step failed
            and ``left_on_stage`` naming what the completed steps created.
        """
        if not self.settings.security.allow_script_execution and is_script_capable_node_type(prim_type):
            return _script_execution_refusal(
                f"Refusing to create a {prim_type!r} prim: that type runs code.", prim_type=prim_type
            )
        if collision is not None and collision not in COLLISION_APPROXIMATIONS:
            return ErrorResponse(
                error=(
                    f"Invalid collision {collision!r}. Must be one of: "
                    f"{', '.join(COLLISION_APPROXIMATIONS)}"
                ),
                error_type="ValueError",
            ).model_dump()
        # Checked before anything is sent: a two-element translation used to
        # raise inside the script after the prim existed, losing the steps
        # report and leaving the prim behind.
        for label, vector in (
            ("translation", translation),
            ("rotation_euler", rotation_euler),
            ("scale", scale),
            ("diffuse_color", diffuse_color),
        ):
            if vector is not None and len(vector) != 3:
                return ErrorResponse(
                    error=f"{label} must have 3 components; got {len(vector)}. Nothing was created.",
                    error_type="ValueError",
                ).model_dump()
        if material_path is None and diffuse_color is not None:
            parent, _, name = prim_path.rpartition("/")
            material_path = f"{parent}/Looks/{name}_Material"

        # Each step names the cores it needs, so a call embeds only those.
        plan: List[Tuple[str, str, Tuple[str, ...]]] = [
            (
                "prim",
                f"_define_prim(stage, prim_path, {_pyval(prim_type)}, {{}})",
                (COERCE_ATTRIBUTE_VALUE_CORE, DEFINE_PRIM_CORE),
            ),
        ]
        if translation is not None or rotation_euler is not None or scale is not None:
            plan.append((
                "transform",
                "_set_prim_transform(stage, prim_path, "
                f"{_pyval(list(translation) if translation is not None else None)}, "
                f"{_pyval(list(rotation_euler) if rotation_euler is not None else None)}, "
                f"{_pyval(list(scale) if scale is not None else None)})",
                (XFORM_READ_CORE, SET_PRIM_TRANSFORM_CORE),
            ))
        if rigid_body:
            plan.append((
                "rigid_body",
                f"_apply_rigid_body(stage, prim_path, {_pyval(bool(kinematic))})",
                (APPLY_RIGID_BODY_CORE,),
            ))
        if collision is not None:
            plan.append((
                "collision",
                f"_apply_collision(stage, prim_path, {_pyval(collision)})",
                (APPLY_COLLISION_CORE,),
            ))
        if mass is not None:
            plan.append((
                "mass",
                f"_set_mass_properties(stage, prim_path, {_pyval(float(mass))}, None, None)",
                (SET_MASS_PROPERTIES_CORE,),
            ))
        if material_path is not None:
            color = [float(c) for c in (diffuse_color or [0.8, 0.8, 0.8])]
            plan.append((
                "material",
                f"_ensure_material(stage, material_path, {_pyval(color)})",
                (DEFINE_MATERIAL_CORE, ENSURE_MATERIAL_CORE),
            ))
            plan.append((
                "bind_material",
                "_bind_material(stage, prim_path, material_path)",
                (BIND_MATERIAL_CORE,),
            ))

        result: Dict[str, Any] = {"prim_path": prim_path, "steps": {}}
        batches = _plan_batches(plan, prim_path, material_path)
        for index, (script, step_names) in enumerate(batches):
            # The paused-edit check reads the timeline once, on the first script.
            annotations = ("paused_edit",) if index == 0 else ()
            outcome = await self._execute_json_script(script, annotations=annotations)
            result["steps"].update(outcome.get("steps") or {})
            for key, value in outcome.items():
                if key in ("steps", "prim_path", "success", "error", "failed_step"):
                    continue
                if key == "warnings" and isinstance(value, list):
                    result.setdefault("warnings", []).extend(value)
                else:
                    result.setdefault(key, value)
            if outcome.get("error") is not None or outcome.get("success") is False:
                result["error"] = outcome.get("error") or "Script failed"
                # A failure before the driver ran (a transport error) names the
                # batch's first step that did not complete.
                result["failed_step"] = outcome.get("failed_step") or next(
                    (name for name in step_names if name not in result["steps"]), step_names[0]
                )
                break
        if len(batches) > 1:
            result["round_trips"] = len(batches)
        if "error" in result:
            created = []
            if "prim" in result["steps"]:
                created.append(prim_path)
            if (result["steps"].get("material") or {}).get("created"):
                created.append(material_path)
            if created:
                result["left_on_stage"] = created
                result["note"] = (
                    "The steps before the failure are kept: " + ", ".join(created)
                    + " exist. Remove them with delete_isaac_prim, or finish the object with "
                    "the single-step tools."
                )
        result["success"] = "error" not in result
        return result


def _plan_batches(
    plan: List[Tuple[str, str, Tuple[str, ...]]],
    prim_path: str,
    material_path: Optional[str],
) -> List[Tuple[str, List[str]]]:
    """Group the plan's steps into as few scripts as fit OBJECT_SCRIPT_BUDGET.

    Steps keep their order; a step starts a new script only when adding it
    would take the current one over the budget, so a typical object is one
    round trip and the full set of steps two.

    Args:
        plan: ``(step name, call source, cores the call needs)`` in order.
        prim_path: The prim the steps build.
        material_path: The material the material steps use, or None.

    Returns:
        ``(script, step names)`` per round trip.
    """
    batches: List[Tuple[str, List[str]]] = []
    current: List[Tuple[str, str, Tuple[str, ...]]] = []
    for step in plan:
        candidate = current + [step]
        if current and len(_object_script(candidate, prim_path, material_path).encode()) > OBJECT_SCRIPT_BUDGET:
            batches.append((_object_script(current, prim_path, material_path), [s[0] for s in current]))
            candidate = [step]
        current = candidate
    batches.append((_object_script(current, prim_path, material_path), [s[0] for s in current]))
    return batches


def _object_script(
    steps: List[Tuple[str, str, Tuple[str, ...]]],
    prim_path: str,
    material_path: Optional[str],
) -> str:
    """Compose the script that runs ``steps`` in order and reports each one.

    Args:
        steps: ``(step name, call source, cores the call needs)`` to run.
        prim_path: The prim the steps build.
        material_path: The material the material steps use, or None.

    Returns:
        The generated script.
    """
    cores: List[str] = []
    for _name, _call, needed in steps:
        for core in needed:
            if core not in cores:
                cores.append(core)
    # Built at column zero rather than inside a dedented template: the
    # generated list lines would otherwise decide the template's common
    # indent and shift every other line.
    plan_source = "".join(
        f"    ({_pyval(step_name)}, lambda: {call}),\n" for step_name, call, _needed in steps
    )
    driver_header = (
        "stage = omni.usd.get_context().get_stage()\n"
        f"prim_path = {_pyval(prim_path)}\n"
        f"material_path = {_pyval(material_path)}\n"
        f"plan = [\n{plan_source}]\n"
    )
    return _compose_script(
        """\
        import json
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf
        """,
        *cores,
        driver_header,
        """\
        if stage is None:
            print(json.dumps({"error": "No stage is currently open"}))
        else:
            steps = {}
            result = {"prim_path": prim_path, "steps": steps}
            for step_name, run_step in plan:
                try:
                    outcome = run_step()
                except Exception as exc:
                    # A step that raises still gets the report of what the
                    # steps before it left on the stage.
                    outcome = {"error": type(exc).__name__ + ": " + str(exc).strip()}
                if "error" in outcome:
                    result["error"] = outcome["error"]
                    result["failed_step"] = step_name
                    for key in ("error_type", "details"):
                        if key in outcome:
                            result[key] = outcome[key]
                    break
                steps[step_name] = outcome
            print(json.dumps(result))
        """,
    )
