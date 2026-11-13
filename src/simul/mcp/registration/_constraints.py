"""Machine-readable bounds and choices for tool parameters.

Many tool parameters had limits that lived only in prose ("1–3840", "one of
DomeLight, DistantLight, ...") or in a silent clamp inside the tool
(``max(1, min(max_results, 1000))``), so an agent could not read them from the
input schema and an out-of-range value was quietly replaced. Two sources now
put them in the schema, where the MCP layer enforces them and refuses an
out-of-range value with a ``ValidationError``:

- ``PARAM_CONSTRAINTS`` below, for tools whose limits are in their code or
  prose. The numbers are the tools' own (``tests/mcp/test_param_constraints.py``
  checks them against the runtime constants and the clamps in the source).
- ``from_request_model``, which copies the bounds, lengths and choices a
  tool's request model already enforces onto the parameter it advertises.

The tools keep their own checks and clamps for the CLI, which calls them
directly. Pure data: the registration helpers import this module while the
Isaac tool modules are still loading, so it imports none of them.
"""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import annotated_types
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo


@dataclass(frozen=True)
class Bounds:
    """Numeric limits of one parameter (inclusive ``ge``/``le``, exclusive ``gt``/``lt``)."""

    ge: Optional[float] = None
    le: Optional[float] = None
    gt: Optional[float] = None
    lt: Optional[float] = None

    def field(self) -> Any:
        return Field(ge=self.ge, le=self.le, gt=self.gt, lt=self.lt)


@dataclass(frozen=True)
class Choices:
    """The values one string parameter takes.

    Attributes:
        values: The choices, exactly as the tool compares them.
        enforced: The MCP layer refuses any other value. False only
            advertises the choices, for a tool that also accepts aliases.
    """

    values: Tuple[str, ...]
    enforced: bool = True


Constraint = Union[Bounds, Choices]

#: A count of results per page (``max_results = max(1, min(max_results, N))``).
_PAGE = Bounds(ge=1, le=1000)
#: A paging offset (``offset = max(0, offset)``).
_OFFSET = Bounds(ge=0)
#: A material factor documented as 0-1.
_UNIT = Bounds(ge=0.0, le=1.0)

#: Isaac Sim viewport capture edge, ``MAX_CAPTURE_DIMENSION`` in the Isaac tools.
ISAAC_MAX_CAPTURE_DIMENSION = 3840
#: Unreal capture edge (``capture_viewport``) and the largest view aspect ratio
#: (``FOCUS_MAX_ASPECT_RATIO``); the narrowest field of view Unreal renders
#: (``FOCUS_MIN_FOV_DEGREES``).
UNREAL_MAX_CAPTURE_DIMENSION = 16384
#: Longest array preview get_prim_info returns (``MAX_ARRAY_PREVIEW``), and the
#: largest search and subtree pages (``MAX_SEARCH_LIMIT``, ``MAX_SUBTREE_LIMIT``).
USD_MAX_ARRAY_PREVIEW = 10_000
USD_MAX_SEARCH_LIMIT = 5000
USD_MAX_SUBTREE_LIMIT = 1000
UNREAL_MAX_ASPECT_RATIO = 16384.0
#: The largest page an Unreal read returns, and the deepest scene graph.
UNREAL_MAX_PAGE = 5000
UNREAL_MAX_SCENE_DEPTH = 64
UNREAL_MIN_FOV_DEGREES = 0.001
#: Longest force ``apply_force`` keeps on (``FORCE_MAX_DURATION_S``, seconds of
#: physics time) and the largest physics-state page (``PHYSICS_STATE_MAX_RESULTS``).
UNREAL_FORCE_MAX_DURATION_S = 600.0
UNREAL_PHYSICS_STATE_MAX_RESULTS = 1000
#: The robotics analysis's largest page per category (``ROBOTICS_MAX_RESULTS``),
#: and a light's intensity and temperature ranges (``LIGHT_MAX_INTENSITY``,
#: ``LIGHT_MIN_TEMPERATURE_K``, ``LIGHT_MAX_TEMPERATURE_K``).
UNREAL_ROBOTICS_MAX_RESULTS = 500
UNREAL_LIGHT_MAX_INTENSITY = 1e9
UNREAL_LIGHT_MIN_TEMPERATURE_K = 1000.0
UNREAL_LIGHT_MAX_TEMPERATURE_K = 15000.0

PARAM_CONSTRAINTS: Dict[str, Dict[str, Constraint]] = {
    # -- Isaac Sim -------------------------------------------------------------
    "capture_isaac_viewport": {
        "width": Bounds(ge=1, le=ISAAC_MAX_CAPTURE_DIMENSION),
        "height": Bounds(ge=1, le=ISAAC_MAX_CAPTURE_DIMENSION),
    },
    "get_isaac_logs": {"last_n": Bounds(ge=1, le=500)},
    "get_isaac_ui_window": {
        "max_depth": Bounds(ge=1, le=32),
        "max_widgets": Bounds(ge=1, le=2000),
    },
    "step_isaac_simulation": {"num_steps": Bounds(ge=1, le=1000)},
    "read_isaac_aovs": {"num_frames": Bounds(ge=1, le=60)},
    "list_isaac_cameras": {"max_results": _PAGE, "offset": _OFFSET},
    "list_isaac_lights": {"max_results": _PAGE, "offset": _OFFSET},
    "get_isaac_subtree": {"max_results": _PAGE, "offset": _OFFSET},
    "get_isaac_texture_dependencies": {"max_results": _PAGE},
    "get_isaac_selection": {"max_results": _PAGE},
    "find_isaac_prims_in_area": {"max_results": _PAGE},
    "list_isaac_articulations": {"max_results": Bounds(ge=1, le=200)},
    "get_isaac_graph_nodes": {"max_results": _PAGE},
    "list_isaac_graph_node_types": {"max_results": Bounds(ge=1, le=2000)},
    "list_isaac_materials": {"max_results": _PAGE, "offset": _OFFSET},
    "list_isaac_physics_objects": {"max_results": _PAGE, "offset": _OFFSET},
    "list_isaac_aovs": {"max_results": _PAGE},
    "query_isaac_typed_prims": {
        "max_results": Bounds(ge=1, le=2000),
        "offset": _OFFSET,
        "max_array_elements": Bounds(ge=0, le=10000),
    },
    "list_isaac_render_vars": {"max_results": _PAGE},
    "list_isaac_prims": {"max_results": _PAGE, "offset": _OFFSET},
    "search_isaac_prims": {"max_results": _PAGE, "offset": _OFFSET},
    "list_isaac_extensions": {"limit": _PAGE, "offset": _OFFSET},
    "create_isaac_material": {
        "shader_type": Choices(("UsdPreviewSurface", "OmniPBR")),
        "roughness": _UNIT,
        "metallic": _UNIT,
        "opacity": _UNIT,
    },
    "create_isaac_light": {
        "light_type": Choices(
            ("DomeLight", "DistantLight", "SphereLight", "RectLight", "DiskLight", "CylinderLight")
        ),
    },
    "create_isaac_object": {
        "collision": Choices(
            (
                "none",
                "convexHull",
                "convexDecomposition",
                "meshSimplification",
                "boundingSphere",
                "boundingCube",
            )
        ),
    },
    "create_isaac_physics_scene": {"gravity_magnitude": Bounds(gt=0)},
    "add_isaac_collision": {
        "approximation": Choices(
            (
                "none",
                "convexHull",
                "convexDecomposition",
                "meshSimplification",
                "boundingSphere",
                "boundingCube",
            )
        ),
    },
    # The tool upper-cases both before checking, so any case is taken.
    "configure_isaac_physics_scene": {
        "solver_type": Choices(("TGS", "PGS"), enforced=False),
        "broadphase_type": Choices(("GPU", "MBP", "SAP"), enforced=False),
    },
    "set_isaac_joint_drive": {
        "drive_axis": Choices(
            ("angular", "linear", "transX", "transY", "transZ", "rotX", "rotY", "rotZ")
        ),
        "drive_type": Choices(("force", "acceleration")),
    },
    "set_isaac_joint_limits": {
        "limit_axis": Choices(("transX", "transY", "transZ", "rotX", "rotY", "rotZ", "distance")),
    },
    # -- Headless USD ----------------------------------------------------------
    # The limits the USD request models validate (simul.mcp.schemas.usd); the
    # choices (search_type, format) the tools advertise as Literal types.
    "get_prim_info": {"array_preview": Bounds(ge=0, le=USD_MAX_ARRAY_PREVIEW)},
    "get_prim_subtree": {
        "max_depth": Bounds(ge=0, le=64),
        "max_results": Bounds(ge=1, le=USD_MAX_SUBTREE_LIMIT),
        "offset": _OFFSET,
        "array_preview": Bounds(ge=0, le=USD_MAX_ARRAY_PREVIEW),
    },
    "search_prims": {"limit": Bounds(ge=1, le=USD_MAX_SEARCH_LIMIT), "offset": _OFFSET},
    # -- Unreal ----------------------------------------------------------------
    "capture_unreal_viewport": {
        "resolution_x": Bounds(ge=1, le=UNREAL_MAX_CAPTURE_DIMENSION),
        "resolution_y": Bounds(ge=1, le=UNREAL_MAX_CAPTURE_DIMENSION),
    },
    "focus_unreal_on_actor": {
        "fov": Bounds(ge=UNREAL_MIN_FOV_DEGREES, lt=180.0),
        "aspect_ratio": Bounds(ge=1.0 / UNREAL_MAX_ASPECT_RATIO, le=UNREAL_MAX_ASPECT_RATIO),
    },
    # The page the editor-side reads take (``_check_page``) and the scene
    # graph's depth.
    "list_unreal_actors": {"max_results": Bounds(ge=1, le=UNREAL_MAX_PAGE), "offset": _OFFSET},
    "list_unreal_actor_components": {
        "max_results": Bounds(ge=1, le=UNREAL_MAX_PAGE),
        "offset": _OFFSET,
    },
    "query_unreal_scene_graph": {
        "max_depth": Bounds(ge=1, le=UNREAL_MAX_SCENE_DEPTH),
        "max_results": Bounds(ge=1, le=UNREAL_MAX_PAGE),
        "offset": _OFFSET,
    },
    "get_unreal_actor_by_semantic_label": {
        "max_results": Bounds(ge=1, le=UNREAL_MAX_PAGE),
        "offset": _OFFSET,
        "match_mode": Choices(("exact", "contains", "regex")),
    },
    # Advertised only: the request model refuses another mode, through the
    # script tool's own echo (the source by its size) and usage record.
    "execute_unreal_script": {
        "mode": Choices(("ExecuteFile", "EvaluateStatement", "ExecuteStatement"), enforced=False),
    },
    # Matched in any case by the editor, so advertised without refusing the rest.
    "generate_unreal_procedural_scene": {
        "scene_type": Choices(("room", "corridor", "warehouse", "outdoor"), enforced=False),
    },
    "apply_unreal_mesh_boolean": {
        "operation": Choices(("union", "subtract", "intersect"), enforced=False),
    },
    "subdivide_unreal_mesh": {
        "level": Bounds(ge=1, le=4),
        "scheme": Choices(("catmull_clark", "loop", "uniform"), enforced=False),
    },
    "simplify_unreal_mesh": {"target_percentage": Bounds(ge=0.0, le=1.0)},
    "set_unreal_actor_visibility": {"scope": Choices(("all", "editor", "game"))},
    "control_unreal_simulation": {
        "action": Choices(("start", "stop", "pause", "resume", "step")),
    },
    "apply_unreal_force": {"duration": Bounds(ge=0.0, le=UNREAL_FORCE_MAX_DURATION_S)},
    "get_unreal_physics_state": {
        "max_results": Bounds(ge=1, le=UNREAL_PHYSICS_STATE_MAX_RESULTS),
        "offset": _OFFSET,
    },
    "list_unreal_simulation_actors": {"max_results": Bounds(ge=1, le=UNREAL_MAX_PAGE), "offset": _OFFSET},
    "analyze_unreal_scene_for_robotics": {
        "max_results": Bounds(ge=1, le=UNREAL_ROBOTICS_MAX_RESULTS),
        "offset": _OFFSET,
    },
    "set_unreal_light_params": {
        "intensity": Bounds(ge=0.0, le=UNREAL_LIGHT_MAX_INTENSITY),
        "temperature": Bounds(ge=UNREAL_LIGHT_MIN_TEMPERATURE_K, le=UNREAL_LIGHT_MAX_TEMPERATURE_K),
    },
    # The editor also takes aliases (patchbuilder, auto_uv; any case), so the
    # canonical names are advertised without refusing the rest.
    "compute_unreal_mesh_uv": {"method": Choices(("auto", "xatlas", "patch_builder"), enforced=False)},
    "convert_unreal_mesh_format": {
        "target_format": Choices(("static_mesh", "dynamic_mesh"), enforced=False),
    },
    # -- Blender (tools whose wrapper is written out; the forwarding tools take
    # their limits from their request models through from_request_model) ------
    "capture_blender_viewport": {
        "width": Bounds(ge=64, le=4096),
        "height": Bounds(ge=64, le=4096),
        "jpeg_quality": Bounds(ge=1, le=100),
    },
    "capture_blender_viewport_sequence": {
        "step": Bounds(ge=1),
        "max_frames": Bounds(ge=1, le=100),
        "width": Bounds(ge=64, le=4096),
        "height": Bounds(ge=64, le=4096),
        "jpeg_quality": Bounds(ge=1, le=100),
    },
    "setup_blender_rigid_body": {"restitution": _UNIT},
    # Compared upper-cased, so other spellings still work.
    "import_blender_file": {
        "file_format": Choices(("OBJ", "FBX", "GLTF", "USD", "STL", "PLY"), enforced=False),
    },
    "export_blender_file": {
        "file_format": Choices(("OBJ", "FBX", "GLTF", "USD", "STL", "PLY"), enforced=False),
    },
}

# annotated_types constraints a request model's field carries that describe
# the value's range or length (validators, which are code, are not copied).
_COPIED_METADATA = (
    annotated_types.Ge,
    annotated_types.Gt,
    annotated_types.Le,
    annotated_types.Lt,
    annotated_types.MultipleOf,
    annotated_types.MinLen,
    annotated_types.MaxLen,
)


def _optional_of(annotation: Any) -> Tuple[Any, bool]:
    """Split ``Optional[X]`` (or ``X | None``) into ``X`` and True; anything else into itself and False."""
    if typing.get_origin(annotation) in (Union, types.UnionType):
        members = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(members) == 1 and len(members) != len(typing.get_args(annotation)):
            return members[0], True
    return annotation, False


def _same_type(left: Any, right: Any) -> bool:
    """Compare two annotations, ``list[float]`` and ``List[float]`` alike."""
    if left == right:
        return True
    origin = typing.get_origin(left)
    return origin is not None and origin == typing.get_origin(right) and (
        typing.get_args(left) == typing.get_args(right)
    )


def _base(annotation: Any) -> Any:
    """Strip ``Annotated`` metadata."""
    while typing.get_origin(annotation) is Annotated:
        annotation = typing.get_args(annotation)[0]
    return annotation


def apply(annotation: Any, constraint: Constraint) -> Any:
    """Return ``annotation`` carrying ``constraint``.

    Choices replace a ``str`` (or ``Optional[str]``) with a ``Literal`` of the
    values when enforced, and add an advertised ``enum`` otherwise. Bounds are
    attached as pydantic constraints, which apply to the value inside an
    ``Optional``.

    A parameter that is already a ``Literal`` keeps its own choices: choices
    that differ would narrow (or widen) what the tool accepts behind its
    back, so they are refused.

    Raises:
        ValueError: Choices for a ``Literal`` parameter that are not its own.
    """
    if isinstance(constraint, Bounds):
        return Annotated[annotation, constraint.field()]
    inner, optional = _optional_of(_base(annotation))
    if typing.get_origin(inner) is Literal:
        if set(typing.get_args(inner)) != set(constraint.values):
            raise ValueError(
                f"choices {constraint.values!r} would replace the Literal "
                f"{typing.get_args(inner)!r} the parameter already declares"
            )
        return annotation
    if constraint.enforced:
        literal: Any = Literal[constraint.values]
        return Optional[literal] if optional else literal
    return Annotated[annotation, Field(json_schema_extra={"enum": list(constraint.values)})]


def constrain(tool_name: str, param_name: str, annotation: Any) -> Any:
    """Return the parameter's annotation with its ``PARAM_CONSTRAINTS`` entry, if any."""
    constraint = PARAM_CONSTRAINTS.get(tool_name, {}).get(param_name)
    return annotation if constraint is None else apply(annotation, constraint)


def from_request_model(
    model: Optional[type[BaseModel]], param_name: str, annotation: Any
) -> Any:
    """Copy the range, length and choice constraints a request model enforces onto a parameter.

    Only when the model's field has the advertised type (or a ``Literal`` of
    strings for an advertised ``str``), so the schema never claims a type the
    tool does not take.

    Args:
        model: The tool's request model, or None.
        param_name: The parameter, and the model field of the same name.
        annotation: The advertised annotation.

    Returns:
        The annotation, constrained as the model's field is.
    """
    field = (model.model_fields if model is not None else {}).get(param_name)
    if field is None:
        return annotation
    advertised, _ = _optional_of(_base(annotation))
    modelled, _ = _optional_of(field.annotation)
    # A field typed with an annotated alias (``Optional[ColorFactor]``) keeps
    # its bounds in the alias, not in the field's own metadata.
    carried: List[Any] = []
    while typing.get_origin(modelled) is Annotated:
        modelled, *extras = typing.get_args(modelled)
        for extra in extras:
            carried.extend(extra.metadata if isinstance(extra, FieldInfo) else [extra])
    if typing.get_origin(modelled) is Literal and advertised is str:
        values = typing.get_args(modelled)
        if not all(isinstance(value, str) for value in values):
            return annotation
        annotation = apply(annotation, Choices(tuple(values)))
    elif not _same_type(modelled, advertised):
        return annotation
    metadata = [m for m in [*carried, *field.metadata] if isinstance(m, _COPIED_METADATA)]
    if not metadata:
        return annotation
    return Annotated[(annotation, *metadata)]


def describe_constraints(schema: Mapping[str, Any]) -> Dict[str, Any]:
    """Pick the bounds and choices out of one property's JSON schema (``anyOf`` included).

    The choices of a list's items count as the property's (a list of names
    drawn from an enum); the items' bounds do not, since they bound each
    element rather than the property.
    """
    found: Dict[str, Any] = {}
    stack: List[Tuple[Mapping[str, Any], bool]] = [(schema, False)]
    while stack:
        node, in_items = stack.pop()
        keys = ("enum",) if in_items else ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "enum")
        for key in keys:
            if key in node:
                found[key] = node[key]
        stack.extend((option, in_items) for option in node.get("anyOf", []))
        items = node.get("items")
        if isinstance(items, Mapping):
            stack.append((items, True))
    return found
