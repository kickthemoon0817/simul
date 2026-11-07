"""The GeometryScript programs the Unreal mesh tools run inside the editor.

Each tool script is ``args = {...}`` (the caller's data, embedded with repr and
never interpolated as Python source), ``MESH_HELPERS`` and the tool's body. The
body defines ``main()`` and ends with ``run(main, ...)``, which prints exactly
one JSON object: the tool's payload, or a typed error envelope. Everything that
touches a UObject happens inside functions: these scripts run in the editor's
shared ``__main__`` scope, where a package, world or other object bound at
module level outlives the call and is reported as a world leak when the next
map loads.

What every script does the same way:

- It needs the GeometryScripting plugin, loaded at editor start; without it the
  call is refused as ``UnsupportedOperation`` (``details.code``
  ``GEOMETRY_SCRIPTING_DISABLED``) naming ``simul unreal setup``, which enables
  the plugin.
- Actor paths resolve through ``actor_at`` (``_unreal_scripts.ACTOR_HELPERS``):
  a live actor of the level open in the editor, found without loading a map. A
  path that names nothing is ``NotFoundError``; an asset path, or an actor with
  no DynamicMeshComponent (a StaticMeshActor), is a ``ValidationError`` with a
  ``details.code`` and the tool that gets there (``convert_unreal_mesh_format``).
- Every GeometryScript call gets a ``GeometryScriptDebug`` and is checked after
  it returns: GeometryScript reports most failures there and otherwise returns
  normally, leaving the mesh as it was (``ApplyPolygroupCatmullClarkSubD`` on a
  mesh without quad polygroups, UE 5.7.4). An error message fails the tool with
  ``EngineError`` carrying the messages; warnings are returned as ``warnings``.
- Operations run on a copy of the mesh. Only when every call succeeded does the
  copy replace the component's mesh, inside one ``ScopedEditorTransaction``, so
  a failure changes nothing and one Undo reverts exactly one edit. The copy is
  swapped in with ``DynamicMeshComponent.set_dynamic_mesh`` rather than written
  into the existing ``UDynamicMesh``: ``UDynamicMesh::Serialize`` skips the mesh
  data when transacting, so an in-place edit inside a transaction is not undone
  (Undo then reverted the previous transaction instead, the actor's spawn among
  them, UE 5.7.4), while the component's mesh pointer is recorded and restored.
  ``GeometryScript_EditorDynamicMeshUtil``'s tracked-change pair cannot help
  from Python: the change container it returns loses its internal pointer in
  the Python struct copy, and the emit is ignored ("ChangeContainer was not
  active", UE 5.7.4). Actors are spawned inside the same kind of transaction,
  so the spawn and its mesh are one Undo step, and the transaction dirties the
  level.
- Changes are refused while Play In Editor runs (``PIEActive``): Unreal spawns
  no editor actors then, and an edit of the editor world would not reach the
  running game. ``validate_unreal_mesh`` only reads and keeps working.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence

from ._unreal_errors import ENGINE_ERROR, NOT_FOUND_ERROR, VALIDATION_ERROR
from ._unreal_scripts import ACTOR_HELPERS, PIE_ACTIVE

#: ``error_type`` of a call this editor cannot serve as asked: the
#: GeometryScripting plugin is not loaded, or the operation is not implemented.
UNSUPPORTED_OPERATION = "UnsupportedOperation"

#: ``error_type`` of a call refused because the attached editor no longer
#: matches ``simul unreal attach`` (another map or editor instance).
INVALID_STATE = "InvalidState"

#: The error types a mesh script reports for a request it refused as asked.
MESH_REFUSALS = (VALIDATION_ERROR, NOT_FOUND_ERROR, UNSUPPORTED_OPERATION, PIE_ACTIVE)

#: The most triangles ``compute_unreal_mesh_uv`` unwraps with XAtlas. XAtlas
#: crashed the editor (a segfault in xatlas::Generate, packing the charts) on
#: lat-long spheres of 130,048 and 318,400 triangles, and its time grows faster
#: than the triangle count: 3.96 s for 7,936 triangles, 12.96 s for 15,840,
#: 17.4 s for an 18,252-triangle box sphere, 28.6 s for 32,256, 47.2 s for
#: 48,048 and 85.6 s for 64,080, all of which survived (UE 5.7.4, Apple M-series).
#: Above this the call is refused (``MESH_TOO_LARGE_FOR_XATLAS``) or, with
#: method ``auto``, PatchBuilder runs, which unwrapped the 318,400-triangle
#: sphere in 29 s.
XATLAS_MAX_TRIANGLES = 20_000

#: The UV methods ``compute_unreal_mesh_uv`` takes.
UV_METHODS = ("auto", "xatlas", "patch_builder")

#: The primitives ``generate_unreal_mesh_primitive`` takes.
PRIMITIVE_TYPES = ("box", "sphere", "cylinder", "cone", "capsule", "torus")

#: The operations ``edit_unreal_mesh_topology`` takes.
TOPOLOGY_OPERATIONS = ("extrude_faces", "bevel_edges", "inset_faces")

#: The formats ``convert_unreal_mesh_format`` takes.
TARGET_FORMATS = ("dynamic_mesh", "static_mesh")

#: The shortest HTTP timeout, in seconds, a mesh script runs under. A
#: GeometryScript call blocks the editor until it returns and Remote Control
#: cannot cancel it, so a call that outlived ``UNREAL__TIMEOUT`` (30 s by
#: default) would keep changing the level after the tool reported a timeout;
#: PatchBuilder needs 29 s for 318,400 triangles.
MESH_SCRIPT_TIMEOUT_S = 120

#: What the plugin refusal suggests.
GEOMETRY_SCRIPTING_HINT = (
    "Run simul unreal setup <project>.uproject --yes, which enables the GeometryScripting plugin "
    "(or enable Geometry Script under Edit > Plugins), then restart the editor: Unreal loads "
    "plugins at startup."
)

#: What a mesh tool refused during Play In Editor suggests.
MESH_PIE_HINT = (
    "Stop PIE with control_unreal_simulation, action 'stop' (full tool surface; CLI: simul unreal sim "
    "stop), then retry. validate_unreal_mesh keeps working while PIE runs; the mesh tools that change "
    "or create meshes do not."
)

#: What a refusal of a StaticMesh actor or asset suggests.
CONVERT_HINT = (
    "convert_unreal_mesh_format with target_format 'dynamic_mesh' and the StaticMesh asset's path "
    "places an editable DynamicMeshActor copy in the level; the mesh tools edit that copy."
)

# Defined before each tool body. ``args`` is the tool's data.
_MESH_PRELUDE = f"""
import math

_GEOMETRY_SCRIPTING_HINT = {GEOMETRY_SCRIPTING_HINT!r}
_CONVERT_HINT = {CONVERT_HINT!r}
_MESH_PIE_HINT = {MESH_PIE_HINT!r}
_XATLAS_MAX_TRIANGLES = {XATLAS_MAX_TRIANGLES!r}
_WARNINGS = []
""" + """
class MeshError(Exception):
    def __init__(self, error_type, message, code=None, **details):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = dict(details)
        if code is not None:
            self.details["code"] = code

    def envelope(self):
        return {"success": False, "error_type": self.error_type, "error": self.message,
                "details": self.details or None}

def refuse(code, message, **details):
    raise MeshError("ValidationError", message, code, **details)

def require_geometry_script():
    if not hasattr(unreal, "GeometryScript_MeshQueries"):
        raise MeshError("UnsupportedOperation",
                        "The Unreal mesh tools need the GeometryScripting plugin, which this editor has not "
                        "loaded", "GEOMETRY_SCRIPTING_DISABLED", hint=_GEOMETRY_SCRIPTING_HINT)

def asset_class(path):
    # An asset path has no subobject part; an actor path does (Map.Map:PersistentLevel.Actor).
    if ":" in path or not path.startswith("/"):
        return None
    data = unreal.EditorAssetLibrary.find_asset_data(path)
    return str(data.asset_class_path.asset_name) if data.is_valid() else None

def mesh_actor(path):
    try:
        return actor_at(path)
    except SimulNotFound as missing:
        kind = asset_class(path)
        if kind == "StaticMesh":
            refuse("STATIC_MESH_ASSET", path + " is a StaticMesh asset, not an actor in the level, and this "
                   "tool edits a DynamicMeshActor's mesh", hint=_CONVERT_HINT)
        if kind is not None:
            refuse("NOT_AN_ACTOR", path + " is a " + kind + " asset, not an actor in the level; give the "
                   "path of a DynamicMeshActor")
        raise MeshError("NotFoundError", str(missing))

def mesh_component(actor):
    root = actor.root_component
    found = [root] if isinstance(root, unreal.DynamicMeshComponent) else list(
        actor.get_components_by_class(unreal.DynamicMeshComponent))
    if len(found) == 1:
        # After an Undo of an earlier edit the component keeps the bounds of the undone mesh
        # (see the module docstring); recompute them from the mesh it now holds.
        found[0].notify_mesh_modified()
        return found[0]
    label = actor.get_actor_label() + " (" + actor.get_class().get_name() + ")"
    if found:
        refuse("AMBIGUOUS_DYNAMIC_MESH", label + " has " + str(len(found)) + " DynamicMeshComponents and "
               "none is its root, so the mesh to edit is ambiguous", actor_path=actor.get_path_name())
    statics = list(actor.get_components_by_class(unreal.StaticMeshComponent))
    if statics:
        mesh = statics[0].get_editor_property("static_mesh")
        refuse("STATIC_MESH_ACTOR", label + " renders a StaticMesh asset, and this tool edits a "
               "DynamicMeshActor's mesh", hint=_CONVERT_HINT, actor_path=actor.get_path_name(),
               static_mesh=mesh.get_path_name() if mesh is not None else None)
    refuse("NO_DYNAMIC_MESH", label + " has no DynamicMeshComponent; give the path of a DynamicMeshActor",
           actor_path=actor.get_path_name())

def refuse_during_pie(action):
    if pie_active():
        raise MeshError(_PIE_ACTIVE, _PIE_ERROR.format(action=action), hint=_MESH_PIE_HINT)

def gs(operation, function, *call_args, **call_kwargs):
    debug = unreal.GeometryScriptDebug()
    result = function(*call_args, debug=debug, **call_kwargs)
    errors = []
    for message in debug.messages:
        text = str(message.message)
        if message.message_type == unreal.GeometryScriptDebugMessageType.ERROR_MESSAGE:
            errors.append(text)
        elif text not in _WARNINGS:
            _WARNINGS.append(text)
    if errors:
        raise MeshError("EngineError", "GeometryScript " + operation + " failed: " + "; ".join(errors)
                        + ". Nothing was changed in the level.", "GEOMETRY_SCRIPT_FAILED",
                        operation=operation, messages=errors)
    return result

def first(value):
    # UFUNCTIONs with out-parameters return (value, *outs).
    return value[0] if isinstance(value, tuple) else value

def triangle_count(mesh):
    return mesh.get_triangle_count()

def vertex_count(mesh):
    return first(unreal.GeometryScript_MeshQueries.get_vertex_count(mesh))

def volume_of(mesh):
    return float(unreal.GeometryScript_MeshQueries.get_mesh_volume_area(mesh)[1])

def working_copy(mesh):
    copy = unreal.DynamicMesh()
    gs("copy", unreal.GeometryScript_MeshDecomposition.copy_mesh_to_mesh, mesh, copy)
    return copy

def commit(actor, component, mesh, description):
    with unreal.ScopedEditorTransaction(description):
        actor.modify()
        component.modify()
        component.set_dynamic_mesh(mesh)

def spawn_mesh_actor(label, transform, mesh, description):
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    with unreal.ScopedEditorTransaction(description) as transaction:
        actor = actors.spawn_actor_from_class(unreal.DynamicMeshActor, transform.translation,
                                              transform.rotation.rotator())
        if actor is None:
            transaction.cancel()
            raise MeshError("EngineError", "Unreal did not spawn a DynamicMeshActor; nothing was changed")
        try:
            actor.set_actor_transform(transform, False, False)
            actor.set_actor_label(label)
            actor.dynamic_mesh_component.set_dynamic_mesh(mesh)
        except Exception:
            actors.destroy_actor(actor)
            transaction.cancel()
            raise
    return actor

def location_only(actor):
    transform = unreal.Transform()
    transform.translation = actor.get_actor_location()
    return transform

def run(main, action=None):
    del _WARNINGS[:]
    try:
        require_geometry_script()
        if action is not None:
            refuse_during_pie(action)
        result = main()
        result["warnings"] = list(_WARNINGS)
    except MeshError as refusal:
        result = refusal.envelope()
    print(json.dumps(result))
"""

#: The helpers every mesh script starts with, after ``args``.
MESH_HELPERS = ACTOR_HELPERS + _MESH_PRELUDE


def mesh_script(body: str, args: Optional[Dict[str, Any]] = None) -> str:
    """The full editor script of one mesh tool.

    Args:
        body: The tool's body: defines ``main()`` and calls ``run``.
        args: The tool's data, embedded with repr.

    Returns:
        The script.
    """
    return f"args = {dict(args or {})!r}\n" + MESH_HELPERS + body


# ---------------------------------------------------------------------------
# Tool bodies
# ---------------------------------------------------------------------------

# Expects args: primitive_type, dimensions (floats), segments, location, label.
GENERATE_PRIMITIVE = """
def append_primitive(mesh):
    prims = unreal.GeometryScript_Primitives
    options = unreal.GeometryScriptPrimitiveOptions()
    options.polygroup_mode = unreal.GeometryScriptPrimitivePolygroupMode.PER_FACE
    xf = unreal.Transform()
    origin = unreal.GeometryScriptPrimitiveOriginMode.CENTER
    d = args["dimensions"]
    seg = max(args["segments"], 4)
    kind = args["primitive_type"]
    if kind == "box":
        gs("append_box", prims.append_box, mesh, options, xf, d.get("width", 100.0), d.get("height", 100.0),
           d.get("depth", 100.0), 1, 1, 1, origin)
    elif kind == "sphere":
        gs("append_sphere_lat_long", prims.append_sphere_lat_long, mesh, options, xf, d.get("radius", 50.0),
           seg, seg, origin)
    elif kind == "cylinder":
        gs("append_cylinder", prims.append_cylinder, mesh, options, xf, d.get("radius", 50.0),
           d.get("height", 100.0), seg, 1, True, origin)
    elif kind == "cone":
        gs("append_cone", prims.append_cone, mesh, options, xf, d.get("base_radius", 50.0), 0.0,
           d.get("height", 100.0), seg, 1, True, origin)
    elif kind == "capsule":
        h = d.get("hemisphere_height", 50.0)
        gs("append_capsule", prims.append_capsule, mesh, options, xf, d.get("radius", 50.0), h, h, seg,
           max(seg // 2, 4), origin)
    else:
        raise MeshError("UnsupportedOperation", "primitive_type 'torus' is not supported yet; nothing was "
                        "created", "NOT_IMPLEMENTED")

def main():
    mesh = unreal.DynamicMesh()
    append_primitive(mesh)
    transform = unreal.Transform()
    transform.translation = unreal.Vector(*args["location"])
    actor = spawn_mesh_actor(args["label"], transform, mesh, "simul: generate mesh " + args["primitive_type"])
    return {"actor_path": actor.get_path_name(), "primitive_type": args["primitive_type"],
            "triangle_count": triangle_count(mesh), "vertex_count": vertex_count(mesh)}

run(main, "spawn actors")
"""

# Expects args: target, tool, operation.
MESH_BOOLEAN = """
OPERATIONS = {"union": "UNION", "subtract": "SUBTRACT", "intersect": "INTERSECTION"}

def main():
    target = mesh_actor(args["target"])
    tool = mesh_actor(args["tool"])
    component = mesh_component(target)
    work = working_copy(component.get_dynamic_mesh())
    before = triangle_count(work)
    operation = getattr(unreal.GeometryScriptBooleanOperation,
                        OPERATIONS.get(args["operation"].lower(), "UNION"))
    gs("apply_mesh_boolean", unreal.GeometryScript_MeshBooleans.apply_mesh_boolean, work, unreal.Transform(),
       mesh_component(tool).get_dynamic_mesh(), unreal.Transform(), operation,
       unreal.GeometryScriptMeshBooleanOptions())
    commit(target, component, work, "simul: mesh boolean " + args["operation"])
    return {"target_mesh_path": target.get_path_name(), "operation": args["operation"],
            "previous_triangle_count": before, "result_triangle_count": triangle_count(work),
            "result_vertex_count": vertex_count(work)}

run(main, "edit meshes")
"""

# Expects args: mesh.
CONVEX_HULL = """
def main():
    source = mesh_actor(args["mesh"])
    mesh = mesh_component(source).get_dynamic_mesh()
    hull = unreal.DynamicMesh()
    gs("compute_mesh_convex_hull", unreal.GeometryScript_Containment.compute_mesh_convex_hull, mesh, hull,
       unreal.GeometryScriptMeshSelection(), unreal.GeometryScriptConvexHullOptions())
    if triangle_count(hull) == 0:
        raise MeshError("EngineError", "GeometryScript computed an empty convex hull for "
                        + source.get_actor_label() + "; nothing was created", "EMPTY_RESULT")
    source_volume = volume_of(mesh)
    closed = unreal.GeometryScript_MeshQueries.get_is_closed_mesh(mesh)
    hull_volume = volume_of(hull)
    actor = spawn_mesh_actor(source.get_actor_label() + "_Hull", location_only(source), hull,
                             "simul: convex hull of " + source.get_actor_label())
    return {"mesh_path": source.get_path_name(), "hull_actor_path": actor.get_path_name(),
            "hull_vertex_count": vertex_count(hull), "hull_triangle_count": triangle_count(hull),
            "source_triangle_count": triangle_count(mesh), "hull_volume": hull_volume,
            "source_volume": source_volume if closed else None,
            "volume_ratio": hull_volume / source_volume if closed and source_volume > 0 else None}

run(main, "spawn actors")
"""

# Expects args: mesh, max_hulls, max_vertices_per_hull.
CONVEX_DECOMPOSITION = """
def hull_info(index, part):
    box = unreal.GeometryScript_MeshQueries.get_mesh_bounding_box(part)
    return {"index": index, "vertex_count": vertex_count(part), "triangle_count": triangle_count(part),
            "volume": volume_of(part), "bounds_min": [box.min.x, box.min.y, box.min.z],
            "bounds_max": [box.max.x, box.max.y, box.max.z]}

def main():
    source = mesh_actor(args["mesh"])
    mesh = mesh_component(source).get_dynamic_mesh()
    decomposition = unreal.DynamicMesh()
    options = unreal.GeometryScriptConvexDecompositionOptions()
    options.num_hulls = args["max_hulls"]
    options.simplify_to_face_count = args["max_vertices_per_hull"]
    gs("compute_mesh_convex_decomposition", unreal.GeometryScript_Containment.compute_mesh_convex_decomposition,
       mesh, decomposition, options)
    if triangle_count(decomposition) == 0:
        raise MeshError("EngineError", "GeometryScript computed an empty convex decomposition for "
                        + source.get_actor_label() + "; nothing was created", "EMPTY_RESULT")
    parts = gs("split_mesh_by_components", unreal.GeometryScript_MeshDecomposition.split_mesh_by_components,
               decomposition, None)[1]
    hulls = [hull_info(index, part) for index, part in enumerate(parts)]
    actor = spawn_mesh_actor(source.get_actor_label() + "_Decomp", location_only(source), decomposition,
                             "simul: convex decomposition of " + source.get_actor_label())
    return {"mesh_path": source.get_path_name(), "decomp_actor_path": actor.get_path_name(),
            "hull_count": len(hulls), "hulls": hulls, "total_vertices": vertex_count(decomposition),
            "total_triangles": triangle_count(decomposition)}

run(main, "spawn actors")
"""

# Expects args: mesh, operation, distance, offset, count.
EDIT_TOPOLOGY = """
def apply(work, operation):
    modeling = unreal.GeometryScript_MeshModeling
    selection = unreal.GeometryScriptMeshSelection()
    if operation == "extrude_faces":
        options = unreal.GeometryScriptMeshLinearExtrudeOptions()
        options.distance = args["distance"]
        gs("apply_mesh_linear_extrude_faces", modeling.apply_mesh_linear_extrude_faces, work, options, selection)
    elif operation == "bevel_edges":
        options = unreal.GeometryScriptMeshBevelSelectionOptions()
        options.bevel_distance = args["distance"]
        options.subdivisions = args["count"]
        gs("apply_mesh_bevel_selection", modeling.apply_mesh_bevel_selection, work, selection,
           unreal.GeometryScriptMeshBevelSelectionMode.TRIANGLE_AREA, options)
    else:
        options = unreal.GeometryScriptMeshInsetOutsetFacesOptions()
        options.distance = args["offset"]
        gs("apply_mesh_inset_outset_faces", modeling.apply_mesh_inset_outset_faces, work, options, selection)

def main():
    actor = mesh_actor(args["mesh"])
    component = mesh_component(actor)
    work = working_copy(component.get_dynamic_mesh())
    before = triangle_count(work)
    apply(work, args["operation"])
    commit(actor, component, work, "simul: mesh " + args["operation"])
    return {"mesh_path": actor.get_path_name(), "operation": args["operation"],
            "previous_triangle_count": before, "result_triangle_count": triangle_count(work),
            "result_vertex_count": vertex_count(work)}

run(main, "edit meshes")
"""

# Expects args: mesh, level, scheme.
SUBDIVIDE = """
def main():
    actor = mesh_actor(args["mesh"])
    component = mesh_component(actor)
    work = working_copy(component.get_dynamic_mesh())
    before = triangle_count(work)
    level = max(1, min(args["level"], 4))
    if args["scheme"].lower() == "catmull_clark":
        gs("apply_polygroup_catmull_clark_sub_d", unreal.GeometryScript_OpenSubdiv.apply_polygroup_catmull_clark_sub_d,
           work, level, unreal.GeometryScriptGroupLayer())
    else:
        gs("apply_uniform_tessellation", unreal.GeometryScript_MeshSubdivide.apply_uniform_tessellation, work, level)
    commit(actor, component, work, "simul: subdivide mesh")
    return {"mesh_path": actor.get_path_name(), "level": level, "scheme": args["scheme"],
            "previous_triangle_count": before, "result_triangle_count": triangle_count(work),
            "result_vertex_count": vertex_count(work)}

run(main, "edit meshes")
"""

# Expects args: mesh, target_triangle_count, target_percentage.
SIMPLIFY = """
def main():
    actor = mesh_actor(args["mesh"])
    component = mesh_component(actor)
    work = working_copy(component.get_dynamic_mesh())
    before = triangle_count(work)
    if args["target_triangle_count"] is not None:
        target = args["target_triangle_count"]
    elif args["target_percentage"] is not None:
        target = max(1, int(before * args["target_percentage"]))
    else:
        target = max(1, before // 2)
    gs("apply_simplify_to_triangle_count", unreal.GeometryScript_MeshSimplification.apply_simplify_to_triangle_count,
       work, target, unreal.GeometryScriptSimplifyMeshOptions())
    commit(actor, component, work, "simul: simplify mesh")
    after = triangle_count(work)
    return {"mesh_path": actor.get_path_name(), "original_triangles": before, "result_triangles": after,
            "result_vertex_count": vertex_count(work),
            "reduction_ratio": round(1.0 - after / before, 4) if before > 0 else 0.0}

run(main, "edit meshes")
"""

# Expects args: mesh, origin, normal, fill_holes. The plane is in the mesh's
# local space.
PLANE_CUT = """
def cut(work, fill):
    frame = unreal.Transform()
    frame.translation = unreal.Vector(*args["origin"])
    frame.rotation = unreal.MathLibrary.make_rot_from_z(unreal.Vector(*args["normal"])).quaternion()
    options = unreal.GeometryScriptMeshPlaneCutOptions()
    options.fill_holes = fill
    gs("apply_mesh_plane_cut", unreal.GeometryScript_MeshBooleans.apply_mesh_plane_cut, work, frame, options)

def main():
    actor = mesh_actor(args["mesh"])
    component = mesh_component(actor)
    mesh = component.get_dynamic_mesh()
    work = working_copy(mesh)
    before = triangle_count(work)
    cut(work, args["fill_holes"])
    filled = 0
    if args["fill_holes"]:
        unfilled = working_copy(mesh)
        cut(unfilled, False)
        filled = triangle_count(work) - triangle_count(unfilled)
    commit(actor, component, work, "simul: cut mesh")
    return {"mesh_path": actor.get_path_name(), "pieces": [actor.get_path_name()], "cut_faces_added": filled,
            "previous_triangle_count": before, "result_triangle_count": triangle_count(work),
            "result_vertex_count": vertex_count(work)}

run(main, "edit meshes")
"""

# Expects args: mesh.
VALIDATE = """
def main():
    queries = unreal.GeometryScript_MeshQueries
    actor = mesh_actor(args["mesh"])
    mesh = mesh_component(actor).get_dynamic_mesh()
    open_edges = first(queries.get_num_open_border_edges(mesh))
    open_loops = first(queries.get_num_open_border_loops(mesh))
    has_normals = bool(queries.get_has_triangle_normals(mesh))
    has_gaps = bool(queries.get_has_triangle_id_gaps(mesh))
    checks = {"watertight": open_edges == 0, "compact": not has_gaps, "has_normals": has_normals}
    issues = []
    if open_edges > 0:
        issues.append(str(open_edges) + " open border edges in " + str(open_loops) + " open border loops")
    if has_gaps:
        issues.append("Triangle ID gaps (the mesh is not compact)")
    if not has_normals:
        issues.append("No normals attribute")
    return {"mesh_path": actor.get_path_name(), "is_valid": checks["watertight"] and checks["compact"],
            "checks": checks, "issues": issues, "triangle_count": triangle_count(mesh),
            "vertex_count": vertex_count(mesh), "open_border_edges": open_edges, "open_border_loops": open_loops,
            "connected_components": first(queries.get_num_connected_components(mesh)),
            "has_normals": has_normals}

run(main)
"""

# Expects args: mesh, target_format.
CONVERT = """
def to_dynamic_mesh():
    path = args["mesh"]
    kind = asset_class(path)
    if kind != "StaticMesh":
        try:
            actor = actor_at(path)
        except SimulNotFound:
            actor = None
        if actor is not None:
            refuse("NOT_A_STATIC_MESH_ASSET", path + " is an actor; give the path of the StaticMesh asset it "
                   "renders to place an editable DynamicMeshActor copy of it")
        if kind is not None:
            refuse("NOT_A_STATIC_MESH_ASSET", path + " is a " + kind + " asset, not a StaticMesh")
        raise MeshError("NotFoundError", "StaticMesh asset not found: " + path)
    asset = unreal.EditorAssetLibrary.load_asset(path)
    mesh = unreal.DynamicMesh()
    lod = unreal.GeometryScriptMeshReadLOD()
    gs("copy_mesh_from_static_mesh", unreal.GeometryScript_AssetUtils.copy_mesh_from_static_mesh, asset, mesh,
       unreal.GeometryScriptCopyMeshFromAssetOptions(), lod)
    actor = spawn_mesh_actor(asset.get_name() + "_Dynamic", unreal.Transform(), mesh,
                             "simul: convert " + asset.get_name() + " to a DynamicMeshActor")
    return {"source_path": path, "source_format": "static_mesh", "result_path": actor.get_path_name(),
            "target_format": "dynamic_mesh", "triangle_count": triangle_count(mesh),
            "vertex_count": vertex_count(mesh)}

def main():
    if args["target_format"] == "dynamic_mesh":
        return to_dynamic_mesh()
    raise MeshError("UnsupportedOperation", "Converting a DynamicMeshActor to a StaticMesh asset is not "
                    "implemented yet; nothing was changed", "NOT_IMPLEMENTED")

run(main, "spawn actors")
"""

# Expects args: mesh, mode, target_edge_length, target_triangle_count,
# smoothing_iterations.
REMESH = """
def mean_edge_length(mesh):
    queries = unreal.GeometryScript_MeshQueries
    lists = unreal.GeometryScript_List
    positions = [(p.x, p.y, p.z) for p in lists.convert_vector_list_to_array(
        queries.get_all_vertex_positions(mesh, False)[1])]
    total = 0.0
    count = 0
    for t in lists.convert_triangle_list_to_array(queries.get_all_triangle_indices(mesh, True)[1]):
        a, b, c = positions[t.x], positions[t.y], positions[t.z]
        total += math.dist(a, b) + math.dist(b, c) + math.dist(c, a)
        count += 3
    return total / count if count else 0.0

def main():
    actor = mesh_actor(args["mesh"])
    component = mesh_component(actor)
    work = working_copy(component.get_dynamic_mesh())
    before = triangle_count(work)
    options = unreal.GeometryScriptRemeshOptions()
    options.remesh_iterations = max(1, args["smoothing_iterations"])
    uniform = unreal.GeometryScriptUniformRemeshOptions()
    if args["target_triangle_count"] is not None:
        uniform.target_triangle_count = args["target_triangle_count"]
        uniform.target_type = unreal.GeometryScriptUniformRemeshTargetType.TRIANGLE_COUNT
    else:
        uniform.target_edge_length = args["target_edge_length"] or 5.0
        uniform.target_type = unreal.GeometryScriptUniformRemeshTargetType.TARGET_EDGE_LENGTH
    gs("apply_uniform_remesh", unreal.GeometryScript_Remeshing.apply_uniform_remesh, work, options, uniform)
    commit(actor, component, work, "simul: remesh mesh")
    return {"mesh_path": actor.get_path_name(), "mode": args["mode"], "original_triangles": before,
            "result_triangles": triangle_count(work), "result_vertex_count": vertex_count(work),
            "average_edge_length": mean_edge_length(work)}

run(main, "edit meshes")
"""

# Expects args: mesh, method ("auto", "xatlas" or "patch_builder"), uv_channel.
COMPUTE_UV = """
def unwrap(work, method, channel):
    uvs = unreal.GeometryScript_UVs
    if method == "xatlas":
        gs("auto_generate_x_atlas_mesh_u_vs", uvs.auto_generate_x_atlas_mesh_u_vs, work, channel,
           unreal.GeometryScriptXAtlasOptions())
    else:
        gs("auto_generate_patch_builder_mesh_u_vs", uvs.auto_generate_patch_builder_mesh_u_vs, work, channel,
           unreal.GeometryScriptPatchBuilderOptions())

def coverage(mesh, channel):
    queries = unreal.GeometryScript_MeshQueries
    area = float(first(queries.get_mesh_uv_area(mesh, channel)))
    box = first(queries.get_uv_set_bounding_box(mesh, channel))
    lo_x, lo_y = min(box.min.x, 0.0), min(box.min.y, 0.0)
    hi_x, hi_y = max(box.max.x, 1.0), max(box.max.y, 1.0)
    return area / ((hi_x - lo_x) * (hi_y - lo_y))

def main():
    actor = mesh_actor(args["mesh"])
    component = mesh_component(actor)
    mesh = component.get_dynamic_mesh()
    channel = args["uv_channel"]
    channels = unreal.GeometryScript_MeshQueries.get_num_uv_sets(mesh)
    if channel < 0 or channel >= channels:
        refuse("UV_CHANNEL_MISSING", actor.get_actor_label() + " has " + str(channels) + " UV channel"
               + ("" if channels == 1 else "s") + (" (0 to " + str(channels - 1) + ")" if channels else "")
               + "; uv_channel " + str(channel) + " does not exist", uv_channels=channels)
    triangles = triangle_count(mesh)
    method = args["method"]
    if method == "auto":
        method = "xatlas" if triangles <= _XATLAS_MAX_TRIANGLES else "patch_builder"
    elif method == "xatlas" and triangles > _XATLAS_MAX_TRIANGLES:
        refuse("MESH_TOO_LARGE_FOR_XATLAS", actor.get_actor_label() + " has " + str(triangles) + " triangles, "
               "more than XAtlas unwraps safely (" + str(_XATLAS_MAX_TRIANGLES) + "): XAtlas crashed the editor "
               "on larger meshes. Use method 'patch_builder' (or 'auto', which picks it at this size).",
               triangle_count=triangles, max_triangles=_XATLAS_MAX_TRIANGLES, suggested_method="patch_builder")
    work = working_copy(mesh)
    unwrap(work, method, channel)
    commit(actor, component, work, "simul: compute mesh UVs")
    return {"mesh_path": actor.get_path_name(), "method": method, "requested_method": args["method"],
            "uv_channel": channel, "island_count": first(unreal.GeometryScript_MeshQueries.get_num_uv_islands(
                work, channel)), "coverage_ratio": coverage(work, channel),
            "triangle_count": triangle_count(work), "vertex_count": vertex_count(work)}

run(main, "edit meshes")
"""

#: The ``details.code`` of a failed GeometryScript call (``EngineError``).
GEOMETRY_SCRIPT_FAILED = "GEOMETRY_SCRIPT_FAILED"

# The attachment guard (``_unreal_attach_scripts`` verify) runs before the tool
# script and raises RuntimeError when the editor no longer matches the
# attachment; the script never starts.
_ATTACHMENT_CHANGED = re.compile(
    r"^RuntimeError: ((?:Unreal attachment changed|Attached viewport is unavailable)[^\n]*?)\s*$",
    re.MULTILINE,
)


def invalid_choice(parameter: str, value: str, choices: Sequence[str], code: str) -> Optional[Dict[str, Any]]:
    """The ``ValidationError`` envelope for a value outside ``choices``, or None.

    Args:
        parameter: The parameter's name.
        value: The value given, already lower-cased.
        choices: The values the tool takes.
        code: The envelope's ``details.code``.

    Returns:
        The envelope, or None when ``value`` is one of ``choices``.
    """
    if value in choices:
        return None
    return {
        "success": False,
        "error_type": VALIDATION_ERROR,
        "error": f"Unknown {parameter} {value!r}; use one of: {', '.join(choices)}. Nothing was changed.",
        "details": {"code": code, "choices": list(choices)},
    }


def attachment_refusal(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The ``InvalidState`` envelope of a script the attachment guard stopped, or None.

    Args:
        result: What ``_execute_json_script`` returned for a mesh script.

    Returns:
        The typed refusal when the failure is the guard's, else None.
    """
    if result.get("success") is not False or result.get("error_type") != "ScriptError":
        return None
    match = _ATTACHMENT_CHANGED.search(str(result.get("error") or ""))
    if match is None:
        return None
    return {
        "success": False,
        "error_type": INVALID_STATE,
        "error": match.group(1) + "; nothing was changed",
        "details": {
            "code": "ATTACHMENT_CHANGED",
            "hint": "The attached editor now shows another map or instance: run simul unreal attach again.",
        },
    }


__all__ = [
    "COMPUTE_UV",
    "PRIMITIVE_TYPES",
    "TARGET_FORMATS",
    "TOPOLOGY_OPERATIONS",
    "UV_METHODS",
    "invalid_choice",
    "attachment_refusal",
    "CONVERT",
    "CONVEX_DECOMPOSITION",
    "CONVEX_HULL",
    "EDIT_TOPOLOGY",
    "ENGINE_ERROR",
    "GENERATE_PRIMITIVE",
    "GEOMETRY_SCRIPT_FAILED",
    "INVALID_STATE",
    "MESH_BOOLEAN",
    "MESH_HELPERS",
    "MESH_REFUSALS",
    "MESH_SCRIPT_TIMEOUT_S",
    "PLANE_CUT",
    "REMESH",
    "SIMPLIFY",
    "SUBDIVIDE",
    "UNSUPPORTED_OPERATION",
    "VALIDATE",
    "XATLAS_MAX_TRIANGLES",
    "mesh_script",
]
