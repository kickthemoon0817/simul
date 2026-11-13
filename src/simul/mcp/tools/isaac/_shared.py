"""Shared constants and helpers for the Isaac tool mixins."""

import json
import re
import textwrap
from typing import Annotated, Any, List, Optional

from pydantic import BeforeValidator

from ....logging import get_logger

logger = get_logger(__name__)

# Extensions the MCP server speaks through. Disabling one saws off the
# transport the agent is using and leaves the instance unreachable.
PROTECTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        "khemoo.simul",
        "isaacsim.code_editor.python_server",
        "isaacsim.code_editor.vscode",
    }
)

# Carb settings namespaces that configure those transports: the bridge, 6.0's
# python_server and 5.x's VS Code socket (host, port, auth on 6.0).
PROTECTED_CARB_SETTING_PREFIXES: tuple[str, ...] = (
    "/exts/khemoo.simul/",
    "/exts/isaacsim.code_editor.python_server/",
    "/exts/isaacsim.code_editor.vscode/",
)

# Carb settings under this prefix are saved to the user's Kit configuration
# (Isaac Sim's .kit files set settings.persistent = true) and apply to every
# later session, not just this one.
PERSISTENT_CARB_SETTING_PREFIX = "/persistent/"

# The same namespaces as Carb paths. A write is refused when its key is one of
# these, lies inside one, or is an ancestor of one (/exts, /, ...): Carb's
# set(path, dict) replaces the whole subtree at path, so a write to /exts or
# to /exts/khemoo.simul itself clears every key beneath it, the bridge's
# require_auth and auth_token included.
PROTECTED_CARB_NAMESPACES: tuple[str, ...] = tuple(
    prefix.rstrip("/") for prefix in PROTECTED_CARB_SETTING_PREFIXES
)
PERSISTENT_CARB_NAMESPACE = PERSISTENT_CARB_SETTING_PREFIX.rstrip("/")

# Settings of the Python Scripting Component (omni.kit.scripting on 5.x,
# omni.behavior.scripting on 6.0), refused while script execution is off.
# /app/scripting/ignoreWarningDialog=true makes Kit run the scripts attached
# to an opened stage without asking (5.x script_manager.py _open_stage; Isaac
# Sim 6.0.1's behaviors.py sets it "to enable behavior scripts").
SCRIPTING_CARB_NAMESPACES: tuple[str, ...] = ("/app/scripting",)


def normalize_carb_path(key: str) -> Optional[str]:
    """Return a Carb key as an absolute path with empty segments dropped.

    Args:
        key: A settings key as a caller wrote it (``exts/a//b/``).

    Returns:
        ``/exts/a/b``, ``/`` for the root, or None when a segment is ``.`` or
        ``..``, which Carb would not resolve the way a reader expects.
    """
    segments = [segment for segment in str(key).split("/") if segment]
    if any(segment in (".", "..") for segment in segments):
        return None
    return "/" + "/".join(segments)


def carb_write_paths(key: str, value: Any) -> Optional[List[str]]:
    """List every Carb path a ``set(key, value)`` writes.

    A dict value writes a subtree: each of its keys, recursively, below
    ``key``. The key itself is listed too, since the subtree replaces
    whatever was there.

    Returns:
        The normalized paths, or None when any of them is malformed.
    """
    root = normalize_carb_path(key)
    if root is None:
        return None
    paths = [root]
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            child = carb_write_paths(root.rstrip("/") + "/" + str(child_key), child_value)
            if child is None:
                return None
            paths.extend(child)
    return paths


def carb_path_touches(path: str, namespace: str) -> bool:
    """Whether writing ``path`` changes something in ``namespace``.

    True when ``path`` is the namespace, lies inside it, or is an ancestor of
    it (a subtree write there replaces the namespace).
    """
    if path == "/":
        return True
    return path == namespace or path.startswith(namespace + "/") or namespace.startswith(path + "/")


# Prim paths whose removal empties the scene rather than editing it.
STAGE_ROOT_PATH = "/"
DEFAULT_WORLD_PATH = "/World"


def _coerce_str_to_float_list(value: Any) -> Any:
    """Pydantic ``BeforeValidator``: accept ``'[1, 2, 3]'`` as a list of floats.

    Some MCP clients serialise list arguments as JSON-encoded strings, and
    Pydantic (>= 2.12) doesn't coerce ``str`` to ``list``, so normalise here.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return value
        if isinstance(parsed, list):
            return parsed
    return value


# List of floats that tolerates JSON-string inputs from forgiving MCP clients.
FloatList = Annotated[List[float], BeforeValidator(_coerce_str_to_float_list)]

# Aspect name -> the method that reads it. Fifteen tools took exactly one
# prim_path and differed only in which of these they returned, so they are one
# tool with a selector rather than fifteen the caller has to tell apart.
PRIM_DETAIL_ASPECTS = {
    "info": "get_isaac_prim_info",
    "transform": "get_isaac_prim_transform",
    "ancestors": "get_isaac_prim_ancestors",
    "relationships": "get_isaac_prim_relationships",
    "variants": "get_isaac_prim_variants",
    "bounding_box": "get_isaac_bounding_box",
    "mesh": "get_isaac_mesh_info",
    "light": "get_isaac_light_info",
    "material": "get_isaac_material_info",
    "rigid_body": "get_isaac_rigid_body_info",
    "collision": "get_isaac_collision_info",
    "joint": "get_isaac_joint_info",
    "mass": "get_isaac_mass_properties",
    "animation": "get_isaac_animation_info",
}

# get_isaac_texture_dependencies takes a *root* path and scans for materials
# beneath it, so as a per-prim "aspect" it answers a different question than
# its name promises: on the standard layout (materials under /World/Looks,
# mesh at /World/Cube) asking a mesh for its textures returns nothing. Left
# out until it can take a prim and resolve that prim's bound material.

# Largest raw script accepted for execution, in UTF-8 bytes, the unit the usage
# log's code_bytes reports a script's size in.
MAX_SCRIPT_BYTES = 100_000

# Script budget, in seconds, for operations that routinely outlast the default
# (the socket timeout less one second, 29 s): opening or creating a stage, a
# camera's first render while shaders compile, enabling a large extension. Past
# the budget the bridge interrupts the script while Kit finishes the work, so
# the tool reported failure for work that then completed. Kept under the
# bridge's per-script ceiling (its 120 s request timeout less 5 s) so the
# bridge still interrupts and reports instead of abandoning the request.
HEAVY_SCRIPT_TIMEOUT_SECONDS = 110.0

# step_isaac_simulation's wall-clock budget: the warm-up a stopped timeline
# spends after play() plus a floor of 10 frames per second (Isaac Sim's full
# app is rate-limited to 60 Hz, and heavy scenes run slower), never below the
# default script budget and never above the heavy one. Stepping stops and
# reports how far it got when the budget runs out.
STEP_BUDGET_BASE_SECONDS = 10.0
STEP_BUDGET_SECONDS_PER_FRAME = 0.1

# Script core shared by the step tool: the rates that decide how many physics
# steps one frame runs. PhysX consumes the timeline's time in steps of the
# physics scene's dt (Isaac Sim's SimulationContext derives its substeps as
# rendering_dt / physics_dt), so a frame is frame_dt * steps_per_second of
# them. Mirrors _physics_rates in the bridge's service.py.
PHYSICS_RATES_CORE = """\
    def _physics_rates(timeline, frame_dt, time_delta):
        rates = {
            "physics_scene": None,
            "physics_steps_per_second": None,
            "physics_dt": None,
            "rate_source": None,
            "timeline_time_codes_per_second": None,
            "rate_limit_hz": None,
            "physics_steps_per_frame": None,
            "physics_steps_estimated": None,
        }
        try:
            rates["timeline_time_codes_per_second"] = timeline.get_time_codes_per_seconds()
        except Exception:
            pass
        try:
            import carb.settings
            settings = carb.settings.get_settings()
            if settings.get("/app/runLoops/main/rateLimitEnabled"):
                rates["rate_limit_hz"] = settings.get("/app/runLoops/main/rateLimitFrequency")
        except Exception:
            pass
        rate = None
        try:
            import omni.usd
            from pxr import PhysxSchema, Usd, UsdPhysics
            stage = omni.usd.get_context().get_stage()
            prims = iter(Usd.PrimRange(stage.GetPseudoRoot())) if stage is not None else iter(())
            for prim in prims:
                # Only the top two levels: /physicsScene, /World/physicsScene.
                if prim.GetPath().pathElementCount >= 2:
                    prims.PruneChildren()
                if not prim.IsA(UsdPhysics.Scene):
                    continue
                rates["physics_scene"] = str(prim.GetPath())
                if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                    rate = PhysxSchema.PhysxSceneAPI(prim).GetTimeStepsPerSecondAttr().Get()
                    rates["rate_source"] = "physxScene:timeStepsPerSecond"
                else:
                    rate = 60.0
                    rates["rate_source"] = "PhysxSceneAPI fallback"
                break
        except Exception:
            pass
        if rate:
            rate = float(rate)
            rates["physics_steps_per_second"] = rate
            rates["physics_dt"] = 1.0 / rate
            if frame_dt:
                rates["physics_steps_per_frame"] = round(frame_dt * rate, 3)
            rates["physics_steps_estimated"] = int(round(time_delta * rate))
        return rates
"""

# 4K on the long edge, so a capture can't grow to tens of megabytes before
# anything checks whether the result can be delivered.
MAX_CAPTURE_DIMENSION = 3840

# Largest capture returned as inline base64. Encoding grows a file by ~33%, and
# ~256 KB of PNG is already a large tool result; past this the caller gets the
# path instead, which is the shape that keeps working.
MAX_INLINE_CAPTURE_BYTES = 262_144

# Captures are kept rather than deleted, since the caller is handed a path — so
# something has to reclaim them. Keep the most recent N and drop the rest: an
# A/B pair needs two, a comparison sweep a few more, and nobody wants the whole
# session's frames sitting in the temp dir.
MAX_RETAINED_CAPTURES = 20

# How much of a Kit log to read when returning its tail. Kit logs reach several
# gigabytes and the read happens on Kit's main thread, so the window bounds the
# freeze; 2 MB still holds far more lines than the 500-entry maximum.
LOG_SCAN_WINDOW_BYTES = 2 * 1024 * 1024

# Array attributes big enough that pulling their value is the cost being
# avoided. Gating on these names rather than on isArray matters: xformOpOrder
# is a one-element token[] telling a caller which xform ops exist, and
# primvars:displayColor is a one-element color3f[] carrying the object colour.
# Both are arrays, neither is bulk, and skipping them loses information the
# caller needs while saving nothing. Anything not listed here takes the normal
# path, which already collapses arrays over 16 elements to a count.
BULK_GEOMETRY_ATTRIBUTES = frozenset(
    {
        "points",
        "normals",
        "velocities",
        "accelerations",
        "faceVertexIndices",
        "faceVertexCounts",
        "holeIndices",
        "cornerIndices",
        "cornerSharpnesses",
        "creaseIndices",
        "creaseLengths",
        "creaseSharpnesses",
        "curveVertexCounts",
        "widths",
        "primvars:st",
        "primvars:normals",
        # PointInstancer
        "positions",
        "orientations",
        "scales",
        "protoIndices",
        "invisibleIds",
        "ids",
    }
)


def _pyval(value: Any) -> str:
    """Render a call parameter as a Python literal for embedding in a script.

    The generated scripts are Python source, so parameters must be embedded as
    Python literals. ``json.dumps`` spells ``None``/``True``/``False`` as
    ``null``/``true``/``false``, which are undefined names once they land in the
    script and raise ``NameError`` inside Isaac Sim. ``repr`` keeps them Python.

    Use this for every embedded parameter, including required ones — a single
    idiom is what keeps the ``null`` class of bug from coming back.
    """
    return repr(value)


def _compose_script(*fragments: str) -> str:
    """Join dedented source fragments into one module-level script.

    The script cores below are plain Python source. Embedding one inside an
    f-string template would indent only its first line, so each fragment is
    dedented on its own and the pieces are concatenated at column zero.

    Full-line comments and blank lines are dropped on the way: they explain
    the cores to whoever reads this module, but Isaac Sim 5.1's stock socket
    takes at most 32 KiB per script, and the cores' comments alone were a
    sixth of create_isaac_object's. Generated scripts hold no multi-line
    string literals, the one place this would change behaviour.

    Args:
        *fragments: Source fragments, each indented however its literal is.

    Returns:
        The concatenated script.
    """
    return "\n".join(_script_source(fragment) for fragment in fragments)


def _script_source(fragment: str) -> str:
    """Return ``fragment`` dedented, without full-line comments or blank lines.

    Args:
        fragment: Python source as written in a script core or template.

    Returns:
        The source as ``_compose_script`` embeds it.
    """
    kept = [
        line
        for line in textwrap.dedent(fragment).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return "\n".join(kept) + "\n"


# Traversal predicate for the listing tools. Isaac robot assets keep their
# meshes and colliders under instanceable prims, where the default predicate
# stops, so a Mesh search or a collider count over a robot came back empty;
# and a deactivated prim (a disabled sensor) never appeared, so "active" was
# always true. Instance proxies are on by default, inactive prims by request.
# The bridge's service carries the same function. Needs ``Usd``.
TRAVERSAL_PREDICATE_CORE = """\
    def _traversal_predicate(include_instance_proxies, include_inactive):
        if include_inactive:
            # An inactive prim also reads as unloaded, so both flags go.
            flags = Usd.PrimIsDefined & ~Usd.PrimIsAbstract
        else:
            flags = Usd.PrimIsActive & Usd.PrimIsDefined & Usd.PrimIsLoaded & ~Usd.PrimIsAbstract
        return Usd.TraverseInstanceProxies(flags) if include_instance_proxies else flags
"""


def _listing_predicate(include_instance_proxies: bool, include_inactive: bool) -> str:
    """Return the script expression that builds a listing's traversal predicate.

    The script must embed ``TRAVERSAL_PREDICATE_CORE``.

    Args:
        include_instance_proxies: Descend into instanceable prims.
        include_inactive: Also visit deactivated prims.

    Returns:
        Python source for the predicate, e.g. ``_traversal_predicate(True, False)``.
    """
    return (
        f"_traversal_predicate({_pyval(bool(include_instance_proxies))}, "
        f"{_pyval(bool(include_inactive))})"
    )


# What the mass and rigid_body aspects say beside a null centre of mass.
# MassAPI's fallback is (-inf, -inf, -inf), the normal state after
# set_isaac_mass_properties(mass=...); json.dumps spells it -Infinity, which
# is not JSON. The bridge's PrimDetailReader carries the same text.
UNSET_CENTER_OF_MASS_NOTE = (
    "Not authored: the schema fallback (-inf) means PhysX derives the centre of mass "
    "from the collision shapes."
)

# Script cores shared by the single-step tools and create_isaac_object. Each
# defines one module-level function that takes the open stage plus plain
# parameters and returns the dict the tool prints; a dict carrying "error"
# reports failure without raising. They are Python *source*, not callables:
# the tools embed them in the generated script so the logic exists once and
# runs inside Kit, whichever tool invoked it.

# Transform readback shared by every tool that reports a prim's pose. The
# bridge extension's PrimDetailReader carries the same logic (it cannot import
# this package inside Kit); test_prim_detail_bridge pins the two together.
# Needs ``Gf`` and ``Usd`` imported by the script.
XFORM_READ_CORE = """\
    def _xf_round(values):
        # Nine decimals drop float noise (6e-17 for zero, 2.0000000000000004)
        # without losing anything a pose needs; + 0.0 turns -0.0 into 0.0.
        return [round(float(value), 9) + 0.0 for value in values]

    def _decompose_matrix(matrix):
        # Matrix4d.ExtractRotation() orthonormalises a matrix that still
        # carries scale, so a rotation under a scaled parent reads wrong (90
        # degrees about Z under a parent scaled by 2 comes back as 60). The
        # first scale op is not the scale either: it misses parents and
        # matrix ops. Gf.Transform factors the scale out first.
        xform = Gf.Transform(matrix)
        rotation = xform.GetRotation()
        quat = rotation.GetQuat()
        # rotateXYZ composes X, then Y, then Z; Decompose takes the axes in
        # the reverse of that order, so read Z, Y, X and flip.
        z_y_x = rotation.Decompose(Gf.Vec3d(0, 0, 1), Gf.Vec3d(0, 1, 0), Gf.Vec3d(1, 0, 0))
        result = {
            "translation": _xf_round(matrix.ExtractTranslation()),
            "rotation_quat": _xf_round([quat.GetReal()] + list(quat.GetImaginary())),
            "rotation_euler_xyz": _xf_round(reversed(z_y_x)),
            "scale": _xf_round(xform.GetScale()),
        }
        # A rotated prim under a non-uniformly scaled parent has axes that
        # are no longer perpendicular: no rotation and per-axis scale
        # reproduce it, so say so rather than present the fit as exact.
        rows = [Gf.Vec3d(matrix[i][0], matrix[i][1], matrix[i][2]) for i in range(3)]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            la = rows[a].GetLength()
            lb = rows[b].GetLength()
            if la > 1e-12 and lb > 1e-12 and abs(Gf.Dot(rows[a], rows[b])) > 1e-6 * la * lb:
                result["sheared"] = True
                break
        return result

    def _transform_report(xformable, world_space, time_code):
        local = _decompose_matrix(xformable.GetLocalTransformation(time_code))
        world = _decompose_matrix(xformable.ComputeLocalToWorldTransform(time_code))
        ops = xformable.GetOrderedXformOps()
        report = {"space": "world" if world_space else "local"}
        report.update(world if world_space else local)
        report["local"] = local
        report["world"] = world
        report["xform_op_order"] = [op.GetOpName() for op in ops]
        animated = [op.GetOpName() for op in ops if op.GetNumTimeSamples() > 0]
        if animated:
            report["time_sampled_ops"] = animated
        if xformable.GetResetXformStack():
            report["resets_xform_stack"] = True
        if local.get("sheared") or world.get("sheared"):
            report["note"] = (
                "The transform is sheared (a rotated prim under a non-uniform scale): "
                "rotation is the closest pure rotation and scale is measured along "
                "the stretch axes, not the prim's own."
            )
        return report
"""

# USD value -> JSON. json.dumps cannot take Gf vectors, quaternions, matrices
# or Vt arrays, and spells an infinite float (the -inf fallback of an unset
# centre of mass, a joint's open limit) as -Infinity, which is not JSON.
# Non-finite floats become null and are counted in _JSON_NON_FINITE so the
# caller can say so. Quaternions read [w, x, y, z], real part first, the
# order set_isaac_prim_attribute accepts. Needs ``Gf`` and ``Sdf``.
JSON_VALUE_CORE = """\
    import math as _jv_math

    _JSON_NON_FINITE = [0]

    def _json_value(value):
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if _jv_math.isfinite(value):
                return value
            _JSON_NON_FINITE[0] += 1
            return None
        if isinstance(value, Sdf.AssetPath):
            return value.path
        if isinstance(value, (Gf.Quatf, Gf.Quatd, Gf.Quath)):
            return [_json_value(float(value.GetReal()))] + [_json_value(float(c)) for c in value.GetImaginary()]
        dimension = getattr(value, "dimension", None)
        if isinstance(dimension, tuple):
            return [[_json_value(float(value[i][j])) for j in range(dimension[1])] for i in range(dimension[0])]
        if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
            try:
                return [_json_value(item) for item in value]
            except Exception:
                pass
        try:
            return _json_value(float(value))
        except Exception:
            return str(value)
"""

# JSON value -> the attribute's USD type. pxr converts a list to neither a
# quaternion nor a matrix (both fail with a bare "type mismatch"), which put
# xformOp:orient, Isaac's default rotation op, and a joint's localRot0/1 out
# of reach; and a build may not convert a list to a Gf vector either (usd-core
# 26.11 refuses [7, 8, 9] for a double3), so every Gf scalar is built
# explicitly and arrays from their converted items. Needs ``Gf``.
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
        if isinstance(dimension, int):
            # A Gf vector (float3, double3, color3f, point3f, texCoord2f, int3, ...).
            if not isinstance(item, (list, tuple)):
                raise ValueError(label + " takes " + str(dimension) + " components as a list")
            if len(item) != dimension:
                raise ValueError(label + " takes " + str(dimension) + " components; got " + str(len(item)))
            return target(*item)
        return item

    def _coerce_attribute_value(attr, value):
        type_name = attr.GetTypeName()
        if type_name.isArray:
            if not isinstance(value, (list, tuple)):
                raise ValueError(str(type_name) + " is an array; pass a list")
            scalar_cls = type_name.scalarType.type.pythonClass
            items = [_coerce_one(item, scalar_cls, str(type_name.scalarType)) for item in value]
            array_cls = type_name.type.pythonClass
            return array_cls(items) if array_cls is not None else items
        return _coerce_one(value, type_name.type.pythonClass, str(type_name))
"""

# A value bound for an asset (or asset[]) attribute is a file Kit reads: a
# texture, an HDRI, an MDL module. Only Kit knows an attribute's type, so the
# server judges every string the caller sent against the file sandbox
# (IsaacScriptBase._asset_value_checks) and the script applies the verdict
# here, once the type is known. A string the server did not check, or any
# other non-empty item, is refused: fail closed. Needs ``Sdf``.
ASSET_SANDBOX_CORE = """\
    def _sandboxed_items(value, checks, strict):
        # Each non-empty string in value (or its list) becomes the path the
        # server resolved for it; returns (value to write, SandboxError or
        # None). strict refuses any other non-empty item too.
        items = value if isinstance(value, list) else [value]
        resolved = []
        for item in items:
            if item == "" or (not strict and not isinstance(item, str)):
                resolved.append(item)  # "" clears the value
                continue
            check = (checks or {}).get(item) if isinstance(item, str) else None
            if check is None or "denial" in check:
                return value, (check or {}).get("denial") or {
                    "success": False,
                    "error": "Not a path the sandbox checked: " + repr(item),
                    "error_type": "SandboxError",
                    "details": {},
                }
            resolved.append(check["path"])
        return (resolved if isinstance(value, list) else resolved[0]), None

    def _sandboxed_asset(type_name, value, checks):
        if type_name not in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
            return value, None
        return _sandboxed_items(value, checks, True)

    def _denied(denial, **context):
        # Where the refused value was going, under details as every route puts it.
        denied = dict(denial)
        denied["details"] = dict(denied.get("details") or {}, **context)
        return denied
"""

# A name (a Carb key's last segment, an OmniGraph input) holds a file location
# when its last word says so: outputDir, /log/file, endpointUrl. "path" alone
# also names prims (robotPath, renderProductPath), so it counts only bare
# (inputs:path) or after a word that makes it a file (usdPath, outputPath).
FILE_LOCATION_LAST_WORDS = frozenset(
    {
        "file", "files", "filename", "filenames", "filepath", "filepaths",
        "dir", "dirs", "directory", "directories", "folder", "folders",
        "url", "urls", "uri", "uris",
    }
)
FILE_PATH_QUALIFIERS = frozenset(
    {
        "file", "usd", "asset", "output", "save", "export", "import", "log", "dump",
        "cache", "config", "data", "root", "search", "extra", "texture", "image",
        "mesh", "model", "script", "dir", "folder", "directory", "temp", "tmp",
    }
)


def names_file_location(name: str) -> bool:
    """Whether ``name`` says its value is a file, folder or URL.

    The last segment of a Carb key (``/log/file``) or an attribute name
    (``inputs:outputDir``) is split into camelCase, ``_`` and ``-`` words and
    only its last word is judged: one of ``FILE_LOCATION_LAST_WORDS``, a bare
    ``path``/``paths``, a ``path`` after one of ``FILE_PATH_QUALIFIERS``, or
    ``fileName``.

    Args:
        name: The key or attribute name.

    Returns:
        True when the name holds a file location.
    """
    last = re.split(r"[/:]", str(name).rstrip("/"))[-1]
    words = [word.lower() for word in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+", last)]
    if not words:
        return False
    *before, final = words
    if final in FILE_LOCATION_LAST_WORDS:
        return True
    if final in ("path", "paths"):
        return not before or before[-1] in FILE_PATH_QUALIFIERS
    if final in ("name", "names"):
        return bool(before) and before[-1] == "file"
    return False


DEFINE_PRIM_CORE = """\
    def _define_prim(stage, prim_path, prim_type, attributes, asset_checks=None):
        # Needs COERCE_ATTRIBUTE_VALUE_CORE and ASSET_SANDBOX_CORE, plus Sdf,
        # Usd and Tf, in the script.
        valid = Sdf.Path.IsValidPathString(prim_path)
        # Returns True, or a (False, reason) tuple, which is truthy.
        if isinstance(valid, tuple):
            valid = valid[0]
        if not valid or not Sdf.Path(prim_path).IsAbsolutePath() or not Sdf.Path(prim_path).IsPrimPath():
            return {
                "error": "Invalid prim path: " + repr(prim_path),
                "error_type": "ValidationError",
                "details": {"code": "INVALID_PATH"},
            }
        existing = stage.GetPrimAtPath(prim_path)
        if existing.IsValid():
            return {
                "error": "Prim already exists: " + prim_path,
                "error_type": "ValidationError",
                "details": {"code": "PRIM_EXISTS"},
            }
        definition = None
        if prim_type:
            # DefinePrim takes any token: a mistyped "cube" made an untyped
            # prim that was reported as created.
            tf_type = Usd.SchemaRegistry.GetTypeFromSchemaTypeName(prim_type)
            if tf_type.isUnknown or not Usd.SchemaRegistry.IsConcrete(tf_type):
                suggestion = None
                for candidate in Tf.Type.Find(Usd.Typed).GetAllDerivedTypes():
                    name = str(Usd.SchemaRegistry.GetSchemaTypeName(candidate))
                    if name.lower() == prim_type.lower() and Usd.SchemaRegistry.IsConcrete(candidate):
                        suggestion = name
                        break
                kind = "an abstract schema" if not tf_type.isUnknown else "not a registered prim type"
                return {
                    "error": (
                        "Unknown prim type " + repr(prim_type) + " (" + kind + ")"
                        + (": did you mean " + repr(suggestion) + "?" if suggestion else ".")
                        + " Use a concrete schema type name such as Xform, Scope, Cube, Mesh or Camera."
                    ),
                    "error_type": "ValidationError",
                    "details": {"code": "UNKNOWN_PRIM_TYPE", "suggestion": suggestion},
                    "suggestion": suggestion,
                }
            definition = Usd.SchemaRegistry().FindConcretePrimDefinition(prim_type)
        # Attributes the schema doesn't define were skipped without a word.
        known = set(definition.GetPropertyNames()) if definition is not None else set()
        unknown = sorted(name for name in attributes if name not in known)
        if unknown:
            return {
                "error": (
                    "Attribute(s) not defined by " + (prim_type or "a typeless prim") + ": "
                    + ", ".join(unknown) + ". Nothing was created; only attributes the type's schema "
                    "defines can be set on creation."
                ),
                "error_type": "ValidationError",
                "details": {"code": "UNKNOWN_ATTRIBUTES", "unknown_attributes": unknown},
                "unknown_attributes": unknown,
            }
        # DefinePrim also defines any missing ancestors. Remember them, so a
        # failure removes everything this call made, not just the leaf.
        # It also turns an over ancestor into a def: remember each one's
        # specifier at the edit target (None: no spec there) to restore it.
        created_ancestors = []
        specifiers = {}
        target = stage.GetEditTarget()
        ancestor = Sdf.Path(prim_path).GetParentPath()
        while not ancestor.IsAbsoluteRootPath():
            existing_ancestor = stage.GetPrimAtPath(ancestor)
            if not existing_ancestor.IsValid():
                created_ancestors.insert(0, str(ancestor))
            elif not existing_ancestor.IsDefined():
                spec = target.GetPrimSpecForScenePath(ancestor)
                specifiers[ancestor] = spec.specifier if spec else None
            ancestor = ancestor.GetParentPath()

        def _undo_define():
            stage.RemovePrim(created_ancestors[0] if created_ancestors else prim_path)
            for path, specifier in specifiers.items():
                spec = target.GetPrimSpecForScenePath(path)
                if spec:
                    spec.specifier = Sdf.SpecifierOver if specifier is None else specifier

        prim = stage.DefinePrim(prim_path, prim_type)
        if not prim.IsValid():
            return {
                "error": "Failed to create prim: " + prim_path,
                "error_type": "InvalidState",
                "details": {"code": "DEFINE_FAILED"},
            }
        for name, value in attributes.items():
            attr = prim.GetAttribute(name)
            denial = None
            if attr.IsValid():
                # An asset value is a file Kit reads: through the sandbox.
                value, denial = _sandboxed_asset(attr.GetTypeName(), value, asset_checks)
            if denial is not None:
                _undo_define()
                return _denied(denial, attribute=name, prim_path=prim_path, created=False)
            try:
                if not attr.IsValid():
                    raise ValueError("it is a relationship, not an attribute")
                if not attr.Set(_coerce_attribute_value(attr, value)):
                    raise ValueError("USD refused the value")
            except Exception as exc:
                # Leave nothing half-built behind a failure.
                _undo_define()
                text = str(exc).strip()
                message = text.splitlines()[-1].split(" : ", 1)[-1].strip(" '") if text else type(exc).__name__
                return {
                    "error": (
                        "Failed to set " + name + " on " + prim_path + ": " + message
                        + "; the prim was not created."
                    ),
                    "error_type": "ValidationError",
                    "details": {
                        "code": "VALUE_REJECTED",
                        "attribute": name,
                        "removed": created_ancestors + [prim_path],
                    },
                }
        result = {
            "prim_path": str(prim.GetPath()),
            "prim_type": prim.GetTypeName(),
            "created": True,
        }
        if created_ancestors:
            result["ancestors_created"] = created_ancestors
        return result
"""

SET_PRIM_TRANSFORM_CORE = """\
    def _set_prim_transform(stage, prim_path, translation, rotation_euler, scale, time_code=None):
        # Input is the prim's *local* transform; the reply reads back both the
        # local and the world pose and labels which is which. Needs
        # XFORM_READ_CORE in the same script.
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"error": "Prim not found: " + prim_path}
        if not prim.IsA(UsdGeom.Xformable):
            return {"error": "Prim is not Xformable: " + prim_path}
        for label, value in (("translation", translation), ("rotation_euler", rotation_euler), ("scale", scale)):
            if value is not None and len(value) != 3:
                return {"error": label + " must have 3 components [x, y, z]; got " + str(len(value))}
        xformable = UsdGeom.Xformable(prim)
        Op = UsdGeom.XformOp
        default_time = Usd.TimeCode.Default()
        when = default_time if time_code is None else Usd.TimeCode(float(time_code))
        rotate_types = (
            Op.TypeRotateXYZ, Op.TypeRotateXZY, Op.TypeRotateYXZ, Op.TypeRotateYZX,
            Op.TypeRotateZXY, Op.TypeRotateZYX, Op.TypeRotateX, Op.TypeRotateY,
            Op.TypeRotateZ, Op.TypeOrient,
        )
        ops = list(xformable.GetOrderedXformOps())
        extra = {}

        def _animated(op):
            return op.GetAttr().GetNumTimeSamples() > 0

        # The prim's own translate / rotate / scale are the unsuffixed,
        # non-inverse ops; pivots and other suffixed ops are left untouched.
        def _is_primary(op, types):
            return (
                not op.IsInverseOp()
                and op.GetOpType() in types
                and len(op.GetOpName().split(":")) == 2
            )

        def _primary(types):
            for index, op in enumerate(ops):
                if _is_primary(op, types):
                    return index
            return None

        def _set_vec(op, value, at):
            precision = op.GetPrecision()
            if precision == Op.PrecisionDouble:
                op.Set(Gf.Vec3d(*value), at)
            elif precision == Op.PrecisionHalf:
                op.Set(Gf.Vec3h(*value), at)
            else:
                op.Set(Gf.Vec3f(*value), at)

        def _set_quat(op, quat, at):
            precision = op.GetPrecision()
            if precision == Op.PrecisionDouble:
                op.Set(Gf.Quatd(quat), at)
            elif precision == Op.PrecisionHalf:
                op.Set(Gf.Quath(quat), at)
            else:
                op.Set(Gf.Quatf(quat), at)

        def _euler_quat(euler):
            # rotateXYZ semantics: rotate about X first, then Y, then Z.
            return (
                Gf.Rotation(Gf.Vec3d(1, 0, 0), float(euler[0]))
                * Gf.Rotation(Gf.Vec3d(0, 1, 0), float(euler[1]))
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), float(euler[2]))
            ).GetQuat()

        def _op_for(attr_name, add_op, precision):
            # An attribute left over from an earlier op stack is reused;
            # adding the op again would fail on the existing attribute.
            attr = prim.GetAttribute(attr_name)
            return UsdGeom.XformOp(attr) if attr.IsValid() else add_op(precision)

        def _hidden_error(names):
            return {
                "error": (
                    ", ".join(names) + " on " + prim_path + " is animated (time-sampled): a value "
                    "written at the default time is hidden by the samples. Pass time_code to "
                    "write a sample at that time."
                ),
                "time_sampled_ops": names,
            }

        # A matrix op holds the whole transform; a translate stacked on it
        # composes with the matrix's own translation instead of replacing it.
        # Swap it for the equivalent translate / orient / scale stack first.
        if any(op.GetOpType() == Op.TypeTransform for op in ops):
            leftovers = [prim.GetAttribute(name) for name in ("xformOp:translate", "xformOp:orient", "xformOp:scale")]
            animated = [op.GetOpName() for op in ops if _animated(op)] + [
                attr.GetName() for attr in leftovers if attr.IsValid() and attr.GetNumTimeSamples() > 0
            ]
            if animated:
                return {
                    "error": (
                        prim_path + " has a matrix op (xformOp:transform) in an animated op stack ("
                        + ", ".join(animated) + "); converting it to translate/orient/scale would "
                        "drop the animation. Set xformOp:transform per time code with "
                        "set_isaac_prim_attribute instead."
                    ),
                    "time_sampled_ops": animated,
                }
            local = xformable.GetLocalTransformation(default_time)
            parts = Gf.Transform(local)
            rebuilt = Gf.Transform()
            rebuilt.SetScale(parts.GetScale())
            rebuilt.SetRotation(parts.GetRotation())
            rebuilt.SetTranslation(parts.GetTranslation())
            if not Gf.IsClose(rebuilt.GetMatrix(), local, 1e-6):
                return {
                    "error": (
                        "The matrix op on " + prim_path + " carries shear, which translate/orient/"
                        "scale cannot express, so it was left unchanged. Set xformOp:transform "
                        "with set_isaac_prim_attribute instead."
                    )
                }
            t_op = _op_for("xformOp:translate", xformable.AddTranslateOp, Op.PrecisionDouble)
            r_op = _op_for("xformOp:orient", xformable.AddOrientOp, Op.PrecisionDouble)
            s_op = _op_for("xformOp:scale", xformable.AddScaleOp, Op.PrecisionDouble)
            _set_vec(t_op, parts.GetTranslation(), default_time)
            _set_quat(r_op, parts.GetRotation().GetQuat(), default_time)
            _set_vec(s_op, parts.GetScale(), default_time)
            ops = [t_op, r_op, s_op]
            extra["converted_matrix_op"] = True
            extra["matrix_op_note"] = (
                "The matrix op was replaced by the equivalent translate/orient/scale stack so its "
                "components could be set; xformOp:transform is no longer in xformOpOrder."
            )

        t_index = _primary((Op.TypeTranslate,))
        r_index = _primary(rotate_types)
        s_index = _primary((Op.TypeScale,))

        # Refuse before anything is written: a default value under time
        # samples changes nothing a viewer sees, and replacing an animated
        # rotate op would throw its animation away.
        if time_code is None:
            hidden = []
            if translation is not None and t_index is not None and _animated(ops[t_index]):
                hidden.append(ops[t_index].GetOpName())
            if (
                rotation_euler is not None
                and r_index is not None
                and ops[r_index].GetOpType() in (Op.TypeOrient, Op.TypeRotateXYZ)
                and _animated(ops[r_index])
            ):
                hidden.append(ops[r_index].GetOpName())
            if scale is not None and s_index is not None and _animated(ops[s_index]):
                hidden.append(ops[s_index].GetOpName())
            if hidden:
                return _hidden_error(hidden)
        if rotation_euler is not None:
            keeps = r_index is not None and ops[r_index].GetOpType() in (Op.TypeOrient, Op.TypeRotateXYZ)
            dropped = [
                op.GetOpName() for index, op in enumerate(ops)
                if _is_primary(op, rotate_types) and not (keeps and index == r_index) and _animated(op)
            ]
            if dropped:
                return {
                    "error": (
                        "rotation_euler would replace the animated rotate op(s) " + ", ".join(dropped)
                        + " on " + prim_path + " with an XYZ rotation and drop their animation; "
                        "set those ops with set_isaac_prim_attribute instead."
                    ),
                    "time_sampled_ops": dropped,
                }

        # A missing op is inserted at its canonical slot (translate, then
        # rotate, then scale) rather than appended: an op appended after an
        # existing scale would, for example, scale the translation.
        if translation is not None:
            if t_index is None:
                ops.insert(0, _op_for("xformOp:translate", xformable.AddTranslateOp, Op.PrecisionDouble))
                t_index = 0
                r_index = None if r_index is None else r_index + 1
                s_index = None if s_index is None else s_index + 1
            _set_vec(ops[t_index], translation, when)

        if rotation_euler is not None:
            rotate_op = None if r_index is None else ops[r_index]
            if rotate_op is not None and rotate_op.GetOpType() == Op.TypeOrient:
                _set_quat(rotate_op, _euler_quat(rotation_euler), when)
            elif rotate_op is not None and rotate_op.GetOpType() == Op.TypeRotateXYZ:
                _set_vec(rotate_op, rotation_euler, when)
            else:
                # No rotation yet, or one in another axis order: the value is
                # XYZ Euler, so a rotateXYZ op takes that slot. An op of
                # another order leaves the stack; its attribute stays.
                new_op = _op_for("xformOp:rotateXYZ", xformable.AddRotateXYZOp, Op.PrecisionFloat)
                _set_vec(new_op, rotation_euler, when)
                if r_index is not None:
                    ops[r_index] = new_op
                else:
                    if s_index is not None:
                        r_index = s_index
                        s_index += 1
                    else:
                        r_index = 0 if t_index is None else t_index + 1
                    ops.insert(r_index, new_op)
            # The value is the prim's whole rotation. Any other unsuffixed
            # rotate op (a split rotateX / rotateY / rotateZ stack) would
            # compose on top of it, so those leave the stack too.
            kept = ops[r_index]
            ops = [
                op for index, op in enumerate(ops)
                if index == r_index or not _is_primary(op, rotate_types)
            ]
            r_index = next(index for index, op in enumerate(ops) if op is kept)
            t_index = _primary((Op.TypeTranslate,))
            s_index = _primary((Op.TypeScale,))

        if scale is not None:
            if s_index is None:
                anchor = r_index if r_index is not None else t_index
                s_index = 0 if anchor is None else anchor + 1
                ops.insert(s_index, _op_for("xformOp:scale", xformable.AddScaleOp, Op.PrecisionFloat))
            _set_vec(ops[s_index], scale, when)

        xformable.SetXformOpOrder(ops, xformable.GetResetXformStack())
        report = {"prim_path": prim_path}
        report.update(_transform_report(xformable, False, when))
        report["time_code"] = None if time_code is None else float(time_code)
        report["translation_set"] = translation
        report["rotation_euler_set"] = rotation_euler
        report["scale_set"] = scale
        report.update(extra)
        return report
"""

OPEN_ASSET_CORE = """\
    def _open_asset(path):
        # AddReference accepts a missing or unreadable file silently and
        # composes nothing; opening the layer first makes that a failure.
        try:
            layer = Sdf.Layer.FindOrOpen(path)
        except Exception as exc:
            return "Asset is not a readable USD layer: " + path + " (" + str(exc).strip() + ")"
        if layer is None:
            return "Asset not found: " + path
        return None
"""

APPLY_RIGID_BODY_CORE = """\
    def _apply_rigid_body(stage, prim_path, kinematic):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"error": "Prim not found: " + prim_path}
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return {"error": "RigidBodyAPI already applied: " + prim_path}
        UsdPhysics.RigidBodyAPI.Apply(prim)
        if kinematic:
            UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)
        return {
            "prim_path": prim_path,
            "rigid_body_applied": True,
            "kinematic": kinematic,
        }
"""

# A CollisionAPI makes a shape only on geometry the physics parser knows; on
# an Xform or Scope (an asset root) it is ignored. So the API goes on the
# geometry at or under prim_path, and the approximation, a MeshCollisionAPI
# token, on meshes only: Cube, Sphere and the other implicit shapes always
# collide analytically. Going down a tree, render-purpose (visual-only) and
# guide geometry is skipped unless all_purposes, so a robot's high-detail
# visual meshes do not become colliders beside its collision meshes. An
# instanceable prim's geometry lives in the shared prototype, edited in the
# source asset, so it is skipped and reported. (The explanation lives here,
# not in the script: create_isaac_object's script must fit one 5.x socket read.)
APPLY_COLLISION_CORE = """\
    def _apply_collision(stage, prim_path, approximation, recursive=True, all_purposes=False):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"error": "Prim not found: " + prim_path}
        if prim.IsInstanceProxy():
            return {"error": "Prim is inside an instance and cannot be edited here: " + prim_path}
        shape_types = [
            UsdGeom.Mesh, UsdGeom.Cube, UsdGeom.Sphere, UsdGeom.Capsule, UsdGeom.Cylinder,
            UsdGeom.Cone, UsdGeom.Plane, UsdGeom.Points,
        ]
        for name in ("Capsule_1", "Cylinder_1"):
            if hasattr(UsdGeom, name):
                shape_types.append(getattr(UsdGeom, name))

        def _is_shape(candidate):
            return any(candidate.IsA(shape_type) for shape_type in shape_types)

        targets = []
        skipped_instances = []
        skipped_purpose = []
        if recursive:
            for candidate in Usd.PrimRange(prim):
                if candidate.IsInstance():
                    skipped_instances.append(str(candidate.GetPath()))
                if not _is_shape(candidate):
                    continue
                purpose = UsdGeom.Imageable(candidate).ComputePurpose()
                if not all_purposes and candidate != prim and purpose in (UsdGeom.Tokens.render, UsdGeom.Tokens.guide):
                    skipped_purpose.append({"path": str(candidate.GetPath()), "purpose": str(purpose)})
                    continue
                targets.append(candidate)
        elif _is_shape(prim):
            targets.append(prim)
        if not targets:
            message = (
                "No collision geometry (Mesh, Cube, Sphere, Capsule, Cylinder, Cone, Plane, "
                "Points) at " + ("or under " if recursive else "") + prim_path
            )
            if not recursive:
                message += "; pass recursive=true to apply collision to the geometry below it"
            if skipped_purpose:
                message += (
                    "; " + str(len(skipped_purpose)) + " render- or guide-purpose shape(s) skipped, "
                    "pass all_purposes=true to include them"
                )
            if skipped_instances:
                message += "; instanced geometry is not editable here: " + ", ".join(skipped_instances[:5])
            return {"error": message}

        def _dynamic_body(shape):
            ancestor = shape
            while ancestor.IsValid() and not ancestor.IsPseudoRoot():
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    body = UsdPhysics.RigidBodyAPI(ancestor)
                    kinematic = body.GetKinematicEnabledAttr().Get()
                    enabled = body.GetRigidBodyEnabledAttr().Get()
                    return enabled is not False and not kinematic
                ancestor = ancestor.GetParent()
            return False

        shapes = []
        warnings = []
        for shape in targets:
            path = str(shape.GetPath())
            entry = {
                "path": path,
                "type": shape.GetTypeName(),
                "already_had_collision": shape.HasAPI(UsdPhysics.CollisionAPI),
            }
            UsdPhysics.CollisionAPI.Apply(shape)
            if shape.IsA(UsdGeom.Mesh):
                if approximation is not None:
                    UsdPhysics.MeshCollisionAPI.Apply(shape).CreateApproximationAttr().Set(approximation)
                # Unauthored, the schema fallback is none (the triangle mesh).
                resolved = UsdPhysics.MeshCollisionAPI(shape).GetApproximationAttr().Get()
                entry["approximation"] = str(resolved) if resolved is not None else "none"
                if entry["approximation"] == "none" and _dynamic_body(shape):
                    warnings.append(
                        path + ": a triangle-mesh collider (approximation none) on a dynamic "
                        "rigid body is simulated by PhysX as convexHull; pass convexHull or "
                        "convexDecomposition to choose"
                    )
            else:
                entry["approximation"] = None
            shapes.append(entry)
        max_listed = 200
        result = {
            "prim_path": prim_path,
            "collision_applied": True,
            "approximation": approximation,
            "recursive": recursive,
            "shape_count": len(shapes),
            "shapes": shapes[:max_listed],
            "shapes_truncated": len(shapes) > max_listed,
        }
        if skipped_instances:
            result["skipped_instances"] = skipped_instances[:max_listed]
            warnings.append(
                str(len(skipped_instances)) + " instanceable prim(s) skipped; their geometry "
                "must be given collision in the source asset"
            )
        if skipped_purpose:
            result["skipped_purpose"] = skipped_purpose[:max_listed]
            result["skipped_purpose_count"] = len(skipped_purpose)
        if warnings:
            result["warnings"] = warnings
        return result
"""

SET_MASS_PROPERTIES_CORE = """\
    def _set_mass_properties(stage, prim_path, mass, density, center_of_mass):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"error": "Prim not found: " + prim_path}
        if not prim.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI(prim)
        if mass is not None:
            mass_api.GetMassAttr().Set(mass)
        if density is not None:
            mass_api.GetDensityAttr().Set(density)
        if center_of_mass is not None:
            mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*center_of_mass))
        # The schema fallback is (-inf, -inf, -inf), meaning "derive from the
        # shape"; json.dumps would spell that as -Infinity, which is not JSON.
        com = mass_api.GetCenterOfMassAttr().Get()
        authored = com is not None and all(abs(c) != float("inf") for c in com)
        return {
            "prim_path": prim_path,
            "mass": mass_api.GetMassAttr().Get(),
            "density": mass_api.GetDensityAttr().Get(),
            "center_of_mass": list(com) if authored else None,
        }
"""

DEFINE_MATERIAL_CORE = """\
    def _define_material(stage, material_path, shader_type, diffuse_color, roughness, metallic, opacity):
        existing = stage.GetPrimAtPath(material_path)
        if existing.IsValid():
            return {"error": "Prim already exists at " + material_path}
        mat = UsdShade.Material(stage.DefinePrim(material_path, "Material"))
        shader_path = material_path + "/Shader"
        shader = UsdShade.Shader(stage.DefinePrim(shader_path, "Shader"))
        color = Gf.Vec3f(*diffuse_color)
        if shader_type == "OmniPBR":
            shader.CreateIdAttr("OmniPBR")
            shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
            shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
            shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(opacity < 1.0)
            if opacity < 1.0:
                shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(opacity)
            out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
            mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
        else:
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
            out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            mat.CreateSurfaceOutput().ConnectToSource(out)
        return {
            "material_path": material_path,
            "shader_path": shader_path,
            "shader_type": shader_type,
            "diffuse_color": list(diffuse_color),
            "roughness": roughness,
            "metallic": metallic,
            "opacity": opacity,
            "created": True,
        }
"""

BIND_MATERIAL_CORE = """\
    def _bind_material(stage, prim_path, material_path):
        prim = stage.GetPrimAtPath(prim_path)
        mat_prim = stage.GetPrimAtPath(material_path)
        if not prim.IsValid():
            return {"error": "Prim not found: " + prim_path}
        if not mat_prim.IsValid() or not mat_prim.IsA(UsdShade.Material):
            return {"error": "Material not found: " + material_path}
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(mat_prim))
        return {
            "prim_path": prim_path,
            "material_path": material_path,
            "bound": True,
        }
"""

# The approximation tokens UsdPhysics.MeshCollisionAPI allows; "none" is the
# mesh itself as a triangle mesh.
COLLISION_APPROXIMATIONS: tuple[str, ...] = (
    "none",
    "convexHull",
    "convexDecomposition",
    "meshSimplification",
    "boundingSphere",
    "boundingCube",
)

# Default cap on list-shaped tool results, applied inside the generated script
# so the trim happens before Kit serialises the payload.
DEFAULT_MAX_RESULTS = 200

# Annotations: checks appended to a tool script, after the line printing the
# tool's own JSON, that report a condition about the result rather than the
# result itself. Each prints one line starting with ANNOTATION_MARKER followed
# by {"fields": {...}, "warnings": [...]}; the tool layer strips those lines
# before parsing the result and merges them into it. Each is a function that
# is called and deleted, so the shared script namespace keeps no names, and
# swallows every exception: an annotation must never fail a tool that worked.
ANNOTATION_MARKER = "[simul-annotation] "

# Isaac Sim's SimulationManager switches PhysX to Fabric for GPU physics:
# enable_fabric() enables omni.physx.fabric and sets /physics/updateToUsd and
# /physics/updateVelocitiesToUsd to false (v5.1.0 simulation_manager.py
# set_physics_sim_device / enable_fabric; v6.0.1 set_device). From then on the
# simulation no longer writes poses and velocities back to USD, so every USD
# read below returns the values from before play while the robot moves.
USD_READBACK_STALE_WARNING = (
    "PhysX is writing simulation results to Fabric, not USD "
    "(/physics/updateToUsd or /physics/updateVelocitiesToUsd is false, as Isaac "
    "Sim's SimulationManager sets for GPU physics), so the poses and velocities "
    "read from USD here are the values from before play, not the simulated state. "
    "Read the simulated state through a physics view (omni.physics.tensors, or "
    "isaacsim.core.experimental.prims on 6.0) with execute_isaac_script."
)

# Only velocity write-back off (/physics/updateVelocitiesToUsd false with
# /physics/updateToUsd on): poses in USD follow the simulation, velocities don't.
USD_VELOCITY_STALE_WARNING = (
    "PhysX is not writing velocities back to USD (/physics/updateVelocitiesToUsd is "
    "false), so the velocities read from USD here are the values from before play; "
    "poses are current. Read velocities through a physics view (omni.physics.tensors, "
    "or isaacsim.core.experimental.prims on 6.0) with execute_isaac_script."
)


def _usd_readback_annotation(reads_velocities: bool) -> str:
    """Return the readback check for a tool that reads poses, and maybe velocities.

    A write-back setting is off when it is set and falsy: Kit settings can
    hold 0 as well as False, and an unset key means the default, which is on.
    Poses are stale when /physics/updateToUsd is off; velocities also when
    /physics/updateVelocitiesToUsd is. A pose-only reader is not flagged for
    velocity write-back alone.

    Args:
        reads_velocities: The tool's result includes velocities read from USD.

    Returns:
        The annotation script.
    """
    return f'''\
def _simul_annotate_usd_readback():
    try:
        import json
        import carb.settings
        import omni.timeline
        if omni.timeline.get_timeline_interface().is_stopped():
            return
        settings = carb.settings.get_settings()
        update_to_usd = settings.get("/physics/updateToUsd")
        update_velocities_to_usd = settings.get("/physics/updateVelocitiesToUsd")
        fabric_enabled = None
        try:
            import omni.kit.app
            manager = omni.kit.app.get_app().get_extension_manager()
            fabric_enabled = bool(manager.is_extension_enabled("omni.physx.fabric"))
        except Exception:
            pass
        poses_stale = update_to_usd is not None and not update_to_usd
        velocities_stale = poses_stale or (
            update_velocities_to_usd is not None and not update_velocities_to_usd
        )
        stale = poses_stale or ({reads_velocities!r} and velocities_stale)
        if not stale and not fabric_enabled:
            return
        fields = {{
            "usd_readback": {{
                "update_to_usd": update_to_usd,
                "update_velocities_to_usd": update_velocities_to_usd,
                "physx_fabric_enabled": fabric_enabled,
                "poses_stale": poses_stale,
                "velocities_stale": velocities_stale,
                "stale_under_fabric": stale,
            }},
        }}
        warnings = []
        if stale:
            fields["stale_under_fabric"] = True
            warnings.append({USD_READBACK_STALE_WARNING!r} if poses_stale else {USD_VELOCITY_STALE_WARNING!r})
        print({ANNOTATION_MARKER!r} + json.dumps({{"fields": fields, "warnings": warnings}}))
    except Exception:
        pass


_simul_annotate_usd_readback()
del _simul_annotate_usd_readback
'''


USD_READBACK_ANNOTATION = _usd_readback_annotation(reads_velocities=False)
USD_VELOCITY_READBACK_ANNOTATION = _usd_readback_annotation(reads_velocities=True)

# Isaac Sim's apps set physics.resetOnStop = true (base and full .kit files of
# 5.1.0 and 6.0.1): stopping or resetting restores what the simulation moved to
# its pre-play state. That reverts an edit made to a simulated prim while the
# timeline is paused (step_isaac_simulation leaves it paused) or playing.
PAUSED_EDIT_WARNING = (
    "The timeline is paused (step_isaac_simulation leaves it paused) and "
    "/physics/resetOnStop is on: stopping or resetting the simulation restores the "
    "prims it simulates to their pre-play state, which also reverts edits made now "
    "to those prims. Stop the simulation before editing to keep the edit."
)

PLAYING_EDIT_WARNING = (
    "The timeline is playing and /physics/resetOnStop is on: stopping or resetting "
    "the simulation restores the prims it simulates to their pre-play state, which "
    "also reverts edits made now to those prims (and the simulation may move them "
    "first). Stop the simulation before editing to keep the edit."
)

PAUSED_EDIT_ANNOTATION = f'''\
def _simul_annotate_paused_edit():
    try:
        import json
        import carb.settings
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_stopped():
            return
        reset_on_stop = carb.settings.get_settings().get("/physics/resetOnStop")
        if reset_on_stop is not None and not reset_on_stop:
            return
        playing = timeline.is_playing()
        note = {{
            "fields": {{"timeline_state": "playing" if playing else "paused"}},
            "warnings": [{PLAYING_EDIT_WARNING!r} if playing else {PAUSED_EDIT_WARNING!r}],
        }}
        print({ANNOTATION_MARKER!r} + json.dumps(note))
    except Exception:
        pass


_simul_annotate_paused_edit()
del _simul_annotate_paused_edit
'''

SCRIPT_ANNOTATIONS: dict[str, str] = {
    "usd_readback": USD_READBACK_ANNOTATION,
    "usd_velocity_readback": USD_VELOCITY_READBACK_ANNOTATION,
    "paused_edit": PAUSED_EDIT_ANNOTATION,
}
