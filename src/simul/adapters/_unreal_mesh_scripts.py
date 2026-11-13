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
- The swap has a price Undo shows. Undo and Redo restore the component's mesh
  pointer but not what ``set_dynamic_mesh`` does around it: the component
  stops listening to the mesh it gets back, so an in-place edit of that mesh
  outside simul (a Blueprint, Python GeometryScript, Modeling Mode) no longer
  updates its bounds or what it draws, and its physics keeps the collision
  from before the Undo or Redo (UE 5.7.4). Every script repairs both on each
  actor it touches (``refresh``). And each edit keeps the replaced mesh alive
  for Undo, a full copy that the undo buffer's size limit does not count.
- A result is capped at ``MAX_RESULT_TRIANGLES``: refused before anything runs
  when its size is known up front, otherwise before it is committed. So is a
  result that would be committed after the caller stopped waiting: a mesh
  script gets the time its HTTP call waits, less ``MESH_COMMIT_MARGIN_S``, and
  discards its result instead of committing it later (``TimeoutError``,
  ``details.code`` ``MESH_TIME_BUDGET``).
- Changes are refused while Play In Editor runs (``PIEActive``): Unreal spawns
  no editor actors then, and an edit of the editor world would not reach the
  running game. ``validate_unreal_mesh`` only reads and keeps working.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Dict, Optional, Sequence

from ._unreal_errors import ENGINE_ERROR, NOT_FOUND_ERROR, TIMEOUT_ERROR, VALIDATION_ERROR, UnrealTimeoutError
from ._unreal_scripts import ACTOR_HELPERS, PIE_ACTIVE

#: ``error_type`` of a call this editor cannot serve as asked: the
#: GeometryScripting plugin is not loaded, or the operation is not implemented.
UNSUPPORTED_OPERATION = "UnsupportedOperation"

#: ``error_type`` of a call refused because the attached editor no longer
#: matches ``simul unreal attach`` (another map or editor instance).
INVALID_STATE = "InvalidState"

#: The error types a mesh script reports for a request it refused as asked,
#: or a result it discarded because it came too late.
MESH_REFUSALS = (VALIDATION_ERROR, NOT_FOUND_ERROR, UNSUPPORTED_OPERATION, PIE_ACTIVE, TIMEOUT_ERROR)

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

#: The most triangles a mesh tool leaves in a mesh it makes or edits. The
#: editor holds the result, the replaced mesh Undo restores and the render
#: data in memory, and the time an operation takes grows with its result. A
#: result that would be larger is refused (``RESULT_TOO_LARGE``) with nothing
#: changed: before anything runs when its size is known up front (a
#: subdivision, a remesh, a UV unwrap or topology edit of a mesh already over
#: it), otherwise when the operation has run and before its result replaces
#: anything.
MAX_RESULT_TRIANGLES = 2_000_000

#: How many more triangles than asked a uniform remesh makes: GeometryScript
#: turns the target into an edge length for equilateral triangles, and the
#: remesher's triangles are not. Measured from 1.17 (a box) to 1.38 (a sphere)
#: at 10,000 to 190,000 triangles, 1.43 for 2,000,000 on a 522,000-triangle
#: sphere and 1.57 for 1,333,333 on a 1M-triangle one (UE 5.7.4). A remesh is
#: predicted at this factor; its result is checked against the cap as well.
REMESH_OVERSHOOT = 1.6

#: The largest ``target_triangle_count`` a remesh takes: its predicted result
#: stays under ``MAX_RESULT_TRIANGLES``.
MAX_REMESH_TARGET_TRIANGLES = int(MAX_RESULT_TRIANGLES / REMESH_OVERSHOOT)

#: The ``segments`` range ``generate_unreal_mesh_primitive`` takes; 512 makes
#: a 522,240-triangle sphere.
MIN_SEGMENTS, MAX_SEGMENTS = 3, 512

#: The UV methods ``compute_unreal_mesh_uv`` takes.
UV_METHODS = ("auto", "xatlas", "patch_builder")

#: The primitives ``generate_unreal_mesh_primitive`` takes.
PRIMITIVE_TYPES = ("box", "sphere", "cylinder", "cone", "capsule")

#: The operations ``edit_unreal_mesh_topology`` takes.
TOPOLOGY_OPERATIONS = ("extrude_faces", "bevel_edges", "inset_faces")

#: The formats ``convert_unreal_mesh_format`` takes.
TARGET_FORMATS = ("dynamic_mesh", "static_mesh")

#: The shortest HTTP timeout, in seconds, a mesh script runs under. A
#: GeometryScript call blocks the editor until it returns and Remote Control
#: cannot cancel it, so the limit has to cover the slowest operations on a
#: mesh of ``MAX_RESULT_TRIANGLES``: a PatchBuilder unwrap of a 2M-triangle
#: sphere took 109 s, its convex hull 94 s, simplifying it to 1,000 triangles
#: 83 s, a 16-hull decomposition 64 s and a remesh of a 1M-triangle sphere to
#: 1.77M 99 s (UE 5.7.4, Apple M-series), so 300 s leaves a slower machine
#: room. A script that runs past it discards its result rather than commit it
#: after the caller stopped waiting.
MESH_SCRIPT_TIMEOUT_S = 300

#: Seconds before the HTTP timeout by which a mesh script must commit: the
#: commit, the reply and its trip back fit in it (swapping in a 2M-triangle
#: mesh took 0.01 s).
MESH_COMMIT_MARGIN_S = 10

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

# Defined before each tool body. ``args`` is the tool's data; ``_BUDGET_S`` and
# ``_DEADLINE`` (``mesh_script``) when it must commit by.
_MESH_PRELUDE = f"""
import math
import time

_STARTED = time.monotonic()
_GEOMETRY_SCRIPTING_HINT = {GEOMETRY_SCRIPTING_HINT!r}
_CONVERT_HINT = {CONVERT_HINT!r}
_MESH_PIE_HINT = {MESH_PIE_HINT!r}
_XATLAS_MAX_TRIANGLES = {XATLAS_MAX_TRIANGLES!r}
_MAX_RESULT_TRIANGLES = {MAX_RESULT_TRIANGLES!r}
_REMESH_OVERSHOOT = {REMESH_OVERSHOOT!r}
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
    # An asset path has no subobject part; an actor path does (Map.Map:PersistentLevel.Actor). The asset
    # registry answers while PIE runs, when EditorAssetLibrary.find_asset_data finds nothing (UE 5.7.4).
    if ":" in path or not path.startswith("/"):
        return None
    package, _, name = path.partition(".")
    name = name or package.rpartition("/")[2]
    for data in unreal.AssetRegistryHelpers.get_asset_registry().get_assets_by_package_name(package):
        if str(data.asset_name) == name:
            return str(data.asset_class_path.asset_name)
    return None

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

def package_dirty(actor):
    package = actor.get_outermost().get_name()
    utils = unreal.EditorLoadingAndSavingUtils
    dirty = list(utils.get_dirty_map_packages()) + list(utils.get_dirty_content_packages())
    return any(p.get_name() == package for p in dirty)

def refresh(component):
    # Undo and Redo restore the component's mesh pointer, not what set_dynamic_mesh does around it:
    # the component no longer listens to the mesh it got back, so an in-place edit of it outside simul
    # leaves the bounds and the render stale, and the physics keeps the collision from before (UE 5.7.4).
    # Setting the mesh it holds binds it again; it also renames the mesh and dirties the level, so it is
    # done only where an Undo or Redo has left the level dirty already. Rebuilding the collision dirties
    # nothing.
    if package_dirty(component.get_owner()):
        component.set_dynamic_mesh(component.get_dynamic_mesh())
    component.update_collision(False)

def mesh_component(actor):
    root = actor.root_component
    found = [root] if isinstance(root, unreal.DynamicMeshComponent) else list(
        actor.get_components_by_class(unreal.DynamicMeshComponent))
    if len(found) == 1:
        refresh(found[0])
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

def require_result_size(triangles, what, predicted=False):
    if triangles > _MAX_RESULT_TRIANGLES:
        refuse("RESULT_TOO_LARGE", what + (" would make about " if predicted else " made ") + str(triangles)
               + " triangles, more than the " + str(_MAX_RESULT_TRIANGLES) + " a mesh tool may leave in a mesh; "
               "nothing was changed", triangle_count=triangles, max_triangles=_MAX_RESULT_TRIANGLES,
               predicted=predicted)

def require_input_size(triangles, label):
    # For the operations whose result keeps every triangle they start from (UVs, extrude, inset, bevel).
    if triangles > _MAX_RESULT_TRIANGLES:
        refuse("RESULT_TOO_LARGE", label + " has " + str(triangles) + " triangles, more than the "
               + str(_MAX_RESULT_TRIANGLES) + " a mesh tool may leave in a mesh, and this operation removes "
               "none; simplify it first. Nothing was changed", triangle_count=triangles,
               max_triangles=_MAX_RESULT_TRIANGLES, predicted=True)

def require_time_left():
    # The caller's HTTP call waits _BUDGET_S plus a margin; a result committed later than that would
    # change the level after the tool reported a timeout.
    if _BUDGET_S is None:
        return
    elapsed = time.monotonic() - _STARTED
    if elapsed > _BUDGET_S or (_DEADLINE is not None and time.time() > _DEADLINE):
        raise MeshError("TimeoutError", "The mesh operation finished after the time this call allows for a "
                        "result to be committed (it ran " + str(round(elapsed, 1)) + " s in the editor; the call "
                        "waits " + str(round(_BUDGET_S)) + " s plus a margin, and UNREAL__TIMEOUT raises it), so "
                        "its result was discarded and nothing was changed", "MESH_TIME_BUDGET",
                        elapsed_s=round(elapsed, 1), budget_s=_BUDGET_S)

def before_commit(*meshes):
    # Every refusal that can stop an edit, checked before the level changes.
    for mesh in meshes:
        require_result_size(triangle_count(mesh), "The operation")
    require_time_left()

def commit(actor, component, mesh, description):
    before_commit(mesh)
    with unreal.ScopedEditorTransaction(description):
        actor.modify()
        component.modify()
        component.set_dynamic_mesh(mesh)
    component.update_collision(False)

def spawn_mesh_actor(label, transform, mesh, description):
    before_commit(mesh)
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


def mesh_script(
    body: str,
    args: Optional[Dict[str, Any]] = None,
    *,
    budget_s: Optional[float] = None,
    deadline: Optional[float] = None,
) -> str:
    """The full editor script of one mesh tool.

    Args:
        body: The tool's body: defines ``main()`` and calls ``run``.
        args: The tool's data, embedded with repr.
        budget_s: Seconds the script may run before it commits a result;
            None sets no limit.
        deadline: ``time.time()`` by which it must commit, for an editor on
            this host's clock; None checks only ``budget_s``.

    Returns:
        The script.
    """
    return (
        f"args = {dict(args or {})!r}\n_BUDGET_S = {budget_s!r}\n_DEADLINE = {deadline!r}\n" + MESH_HELPERS + body
    )


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
    seg = args["segments"]
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
    else:
        h = d.get("hemisphere_height", 50.0)
        gs("append_capsule", prims.append_capsule, mesh, options, xf, d.get("radius", 50.0), h, h, seg,
           max(seg // 2, 4), origin)

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
    result = {"target_mesh_path": target.get_path_name(), "operation": args["operation"],
              "previous_triangle_count": before, "result_triangle_count": triangle_count(work),
              "result_vertex_count": vertex_count(work)}
    commit(target, component, work, "simul: mesh boolean " + args["operation"])
    return result

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
    result = {"mesh_path": source.get_path_name(), "hull_vertex_count": vertex_count(hull),
              "hull_triangle_count": triangle_count(hull), "source_triangle_count": triangle_count(mesh),
              "hull_volume": hull_volume, "source_volume": source_volume if closed else None,
              "volume_ratio": hull_volume / source_volume if closed and source_volume > 0 else None}
    actor = spawn_mesh_actor(source.get_actor_label() + "_Hull", location_only(source), hull,
                             "simul: convex hull of " + source.get_actor_label())
    result["hull_actor_path"] = actor.get_path_name()
    return result

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
    result = {"mesh_path": source.get_path_name(), "hull_count": len(hulls), "hulls": hulls,
              "total_vertices": vertex_count(decomposition), "total_triangles": triangle_count(decomposition)}
    actor = spawn_mesh_actor(source.get_actor_label() + "_Decomp", location_only(source), decomposition,
                             "simul: convex decomposition of " + source.get_actor_label())
    result["decomp_actor_path"] = actor.get_path_name()
    return result

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
    before = triangle_count(component.get_dynamic_mesh())
    require_input_size(before, actor.get_actor_label())
    work = working_copy(component.get_dynamic_mesh())
    apply(work, args["operation"])
    result = {"mesh_path": actor.get_path_name(), "operation": args["operation"],
              "previous_triangle_count": before, "result_triangle_count": triangle_count(work),
              "result_vertex_count": vertex_count(work)}
    commit(actor, component, work, "simul: mesh " + args["operation"])
    return result

run(main, "edit meshes")
"""

# Expects args: mesh, level, scheme.
SUBDIVIDE = """
def main():
    actor = mesh_actor(args["mesh"])
    component = mesh_component(actor)
    before = triangle_count(component.get_dynamic_mesh())
    level = max(1, min(args["level"], 4))
    catmull_clark = args["scheme"].lower() == "catmull_clark"
    # Catmull-Clark splits each quad into 4 per level, uniform tessellation each triangle into (level + 1)^2.
    predicted = before * (4 ** level if catmull_clark else (level + 1) ** 2)
    require_result_size(predicted, "Subdividing " + str(before) + " triangles at level " + str(level)
                        + " (use a lower level or simplify first)", predicted=True)
    work = working_copy(component.get_dynamic_mesh())
    if catmull_clark:
        gs("apply_polygroup_catmull_clark_sub_d", unreal.GeometryScript_OpenSubdiv.apply_polygroup_catmull_clark_sub_d,
           work, level, unreal.GeometryScriptGroupLayer())
    else:
        gs("apply_uniform_tessellation", unreal.GeometryScript_MeshSubdivide.apply_uniform_tessellation, work, level)
    result = {"mesh_path": actor.get_path_name(), "level": level, "scheme": args["scheme"],
              "previous_triangle_count": before, "result_triangle_count": triangle_count(work),
              "result_vertex_count": vertex_count(work)}
    commit(actor, component, work, "simul: subdivide mesh")
    return result

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
    after = triangle_count(work)
    result = {"mesh_path": actor.get_path_name(), "original_triangles": before, "result_triangles": after,
              "result_vertex_count": vertex_count(work),
              "reduction_ratio": round(1.0 - after / before, 4) if before > 0 else 0.0}
    commit(actor, component, work, "simul: simplify mesh")
    return result

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
    result = {"mesh_path": actor.get_path_name(), "pieces": [actor.get_path_name()], "cut_faces_added": filled,
              "previous_triangle_count": before, "result_triangle_count": triangle_count(work),
              "result_vertex_count": vertex_count(work)}
    commit(actor, component, work, "simul: cut mesh")
    return result

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
    result = {"source_path": path, "source_format": "static_mesh", "target_format": "dynamic_mesh",
              "triangle_count": triangle_count(mesh), "vertex_count": vertex_count(mesh)}
    actor = spawn_mesh_actor(asset.get_name() + "_Dynamic", unreal.Transform(), mesh,
                             "simul: convert " + asset.get_name() + " to a DynamicMeshActor")
    result["result_path"] = actor.get_path_name()
    return result

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
    mesh = component.get_dynamic_mesh()
    before = triangle_count(mesh)
    options = unreal.GeometryScriptRemeshOptions()
    options.remesh_iterations = max(1, args["smoothing_iterations"])
    uniform = unreal.GeometryScriptUniformRemeshOptions()
    if args["target_triangle_count"] is not None:
        uniform.target_triangle_count = args["target_triangle_count"]
        uniform.target_type = unreal.GeometryScriptUniformRemeshTargetType.TRIANGLE_COUNT
        predicted = int(_REMESH_OVERSHOOT * args["target_triangle_count"])
        what = "Remeshing to " + str(args["target_triangle_count"]) + " triangles"
    else:
        length = args["target_edge_length"] or 5.0
        uniform.target_edge_length = length
        uniform.target_type = unreal.GeometryScriptUniformRemeshTargetType.TARGET_EDGE_LENGTH
        # Equilateral triangles of edge L cover sqrt(3)/4 L^2 each.
        area = float(unreal.GeometryScript_MeshQueries.get_mesh_volume_area(mesh)[0])
        predicted = int(_REMESH_OVERSHOOT * 4.0 * area / (math.sqrt(3.0) * length * length))
        what = "Remeshing " + str(round(area)) + " cm2 at a " + str(length) + " cm edge (use a longer edge)"
    require_result_size(predicted, what, predicted=True)
    work = working_copy(mesh)
    gs("apply_uniform_remesh", unreal.GeometryScript_Remeshing.apply_uniform_remesh, work, options, uniform)
    before_commit(work)
    result = {"mesh_path": actor.get_path_name(), "mode": args["mode"], "original_triangles": before,
              "result_triangles": triangle_count(work), "result_vertex_count": vertex_count(work),
              "average_edge_length": mean_edge_length(work)}
    commit(actor, component, work, "simul: remesh mesh")
    return result

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
    require_input_size(triangles, actor.get_actor_label())
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
    result = {"mesh_path": actor.get_path_name(), "method": method, "requested_method": args["method"],
              "uv_channel": channel, "island_count": first(unreal.GeometryScript_MeshQueries.get_num_uv_islands(
                  work, channel)), "coverage_ratio": coverage(work, channel),
              "triangle_count": triangle_count(work), "vertex_count": vertex_count(work)}
    commit(actor, component, work, "simul: compute mesh UVs")
    return result

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


def refusal(code: str, message: str, **details: Any) -> Dict[str, Any]:
    """A ``ValidationError`` envelope with ``details.code``; nothing was sent to the editor.

    Args:
        code: The envelope's ``details.code``.
        message: What was wrong.
        **details: Further details.

    Returns:
        The envelope.
    """
    return {
        "success": False,
        "error_type": VALIDATION_ERROR,
        "error": message + ". Nothing was changed.",
        "details": {"code": code, **details},
    }


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
    return refusal(code, f"Unknown {parameter} {value!r}; use one of: {', '.join(choices)}", choices=list(choices))


def is_loopback(host: str) -> bool:
    """Whether ``host`` is this machine, whose clock a mesh script's deadline can use.

    Args:
        host: The Remote Control host.

    Returns:
        True for ``localhost`` and loopback addresses.
    """
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def mesh_timeout_error(
    error: UnrealTimeoutError, timeout: float, budget: float, same_clock: bool
) -> UnrealTimeoutError:
    """The timeout of a mesh script, saying what becomes of its result.

    Args:
        error: The transport's timeout.
        timeout: The seconds the call waited.
        budget: The seconds the script had to commit in.
        same_clock: The script checked a deadline on this host's clock, so
            it commits nothing after the call stopped waiting, however long
            it waited in the editor's queue.

    Returns:
        The error to raise, with ``details.code`` ``MESH_TIME_BUDGET``.
    """
    after = (
        "so when the editor finishes this one it discards the result, and the level stays as it was"
        if same_clock
        else "by its own run time, so it discards a result it finishes later; one that first waited behind "
        "another script in the editor may still commit, so check the level before retrying"
    )
    message = (
        f"The editor did not finish the mesh operation within {timeout:g}s. A mesh script commits its "
        f"result only within {budget:g}s, {after}. The editor stays busy until it finishes. Use a smaller "
        f"mesh or target, or raise UNREAL__TIMEOUT (the mesh tools wait at least {MESH_SCRIPT_TIMEOUT_S}s)"
    )
    details = dict(error.details or {})
    details.update(code="MESH_TIME_BUDGET", budget_s=budget)
    return UnrealTimeoutError(message, TIMEOUT_ERROR, details=details)


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
    "MAX_REMESH_TARGET_TRIANGLES",
    "MAX_RESULT_TRIANGLES",
    "MAX_SEGMENTS",
    "MESH_BOOLEAN",
    "MESH_COMMIT_MARGIN_S",
    "MESH_HELPERS",
    "MESH_REFUSALS",
    "MESH_SCRIPT_TIMEOUT_S",
    "MIN_SEGMENTS",
    "PLANE_CUT",
    "REMESH",
    "REMESH_OVERSHOOT",
    "SIMPLIFY",
    "SUBDIVIDE",
    "UNSUPPORTED_OPERATION",
    "VALIDATE",
    "XATLAS_MAX_TRIANGLES",
    "is_loopback",
    "mesh_script",
    "mesh_timeout_error",
    "refusal",
]
