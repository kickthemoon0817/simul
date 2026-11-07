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
- Positions an agent gives are in world space unless a tool says otherwise:
  booleans combine both actors' meshes at their world transforms, a plane cut
  converts a world plane into the mesh's local space, and the actors a tool
  creates from a source mesh take the source's whole transform.
- The tools that only read a mesh or make a new one from it (validation, hull,
  decomposition) also take a StaticMeshActor or a StaticMesh asset: its first
  LOD is copied into a transient DynamicMesh (``read_source``). The tools that
  edit a mesh in place refuse those, naming ``convert_unreal_mesh_format``,
  which turns a StaticMesh into a DynamicMeshActor and a DynamicMeshActor
  into a StaticMesh asset.
- Assets are written only under ``/Game``: a StaticMesh made, overwritten or
  given collision anywhere else (``/Engine``, a plugin's content) is refused
  as ``SandboxError``, and a save writes exactly that asset's package, never
  every dirty one.
- Changes are refused while Play In Editor runs (``PIEActive``): Unreal spawns
  no editor actors then, and an edit of the editor world would not reach the
  running game. ``validate_unreal_mesh`` only reads and keeps working.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._unreal_errors import ENGINE_ERROR, NOT_FOUND_ERROR, SANDBOX_ERROR, VALIDATION_ERROR
from ._unreal_scripts import ACTOR_HELPERS, PIE_ACTIVE

#: ``error_type`` of a call this editor cannot serve as asked: the
#: GeometryScripting plugin is not loaded, or the operation is not implemented.
UNSUPPORTED_OPERATION = "UnsupportedOperation"

#: ``error_type`` of a call refused because the attached editor no longer
#: matches ``simul unreal attach`` (another map or editor instance).
INVALID_STATE = "InvalidState"

#: The error types a mesh script reports for a request it refused as asked.
MESH_REFUSALS = (VALIDATION_ERROR, NOT_FOUND_ERROR, UNSUPPORTED_OPERATION, PIE_ACTIVE, SANDBOX_ERROR)

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

#: The most triangles a subdivision or remesh may produce. Each multiplies the
#: triangle count (Catmull-Clark and Loop by 4 per level, uniform tessellation
#: by (level + 1)^2), and the editor holds the result, its undo copy and its
#: render data in memory; larger requests are refused before anything runs.
MAX_RESULT_TRIANGLES = 2_000_000

#: The UV methods ``compute_unreal_mesh_uv`` takes.
UV_METHODS = ("auto", "xatlas", "patch_builder")

#: The primitives ``generate_unreal_mesh_primitive`` takes, each with its
#: dimensions (cm) and their defaults. The axes are Unreal's: X forward, Y
#: right, Z up; every primitive is centred on the actor.
PRIMITIVE_DIMENSIONS: Dict[str, Dict[str, float]] = {
    "box": {"width": 100.0, "depth": 100.0, "height": 100.0},
    "sphere": {"radius": 50.0},
    "cylinder": {"radius": 50.0, "height": 100.0},
    "cone": {"base_radius": 50.0, "top_radius": 0.0, "height": 100.0},
    "capsule": {"radius": 30.0, "length": 75.0},
    "torus": {"major_radius": 50.0, "minor_radius": 20.0},
}

#: The primitives ``generate_unreal_mesh_primitive`` takes.
PRIMITIVE_TYPES = tuple(PRIMITIVE_DIMENSIONS)

#: The operations ``edit_unreal_mesh_topology`` takes.
TOPOLOGY_OPERATIONS = ("extrude_faces", "inset_faces", "bevel_edges")

#: The subdivision schemes ``subdivide_unreal_mesh`` takes.
SUBDIVISION_SCHEMES = ("catmull_clark", "loop", "uniform")

#: The boolean operations ``apply_unreal_mesh_boolean`` takes, and the
#: GeometryScriptBooleanOperation each runs. ``difference`` is ``subtract``.
BOOLEAN_OPERATIONS = {"union": "UNION", "subtract": "SUBTRACT", "intersect": "INTERSECTION"}
BOOLEAN_ALIASES = {"difference": "subtract", "intersection": "intersect"}

#: Where a hull or decomposition goes: a new actor (with the result as its
#: simple collision), the source's simple collision, or both.
HULL_OUTPUTS = ("actor", "collision", "both")

#: The formats ``convert_unreal_mesh_format`` takes.
TARGET_FORMATS = ("dynamic_mesh", "static_mesh")

#: The simple collision a StaticMesh made by ``convert_unreal_mesh_format``
#: gets: ``auto`` copies the DynamicMeshComponent's simple collision when it
#: has any and otherwise computes one convex hull, ``component`` copies it
#: (none when it has none), ``convex_hull`` computes one, ``none`` leaves the
#: asset without simple collision.
STATIC_MESH_COLLISION = ("auto", "component", "convex_hull", "none")

#: Where ``convert_unreal_mesh_format`` puts a StaticMesh without an
#: ``asset_path``: ``/Game/Generated/SM_<actor label>``.
DEFAULT_ASSET_FOLDER = "/Game/Generated"

#: The checks ``validate_unreal_mesh`` runs, in report order.
VALIDATION_CHECKS = ("closed", "manifold", "degenerate_triangles", "orientation", "compact", "normals", "uvs")

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

#: The next step after writing a StaticMesh without replace_actor.
EXPORT_HINT = (
    "The DynamicMeshActor is still in the level. Unreal's USD exporter writes DynamicMeshActors as empty "
    "Xforms, without their mesh: to export or ship this mesh, place the asset as a StaticMeshActor (run "
    "this again with replace_actor true, or spawn_unreal_actor with the asset)."
)

#: What a failed Catmull-Clark subdivision suggests.
CATMULL_CLARK_HINT = (
    "Catmull-Clark subdivides the mesh's polygroups as polygons, so it needs polygroups that form "
    "quads (primitives made by generate_unreal_mesh_primitive have them). Use scheme 'loop' for a "
    "triangle mesh."
)

# Defined before each tool body. ``args`` is the tool's data.
_MESH_PRELUDE = f"""
import math

_GEOMETRY_SCRIPTING_HINT = {GEOMETRY_SCRIPTING_HINT!r}
_CONVERT_HINT = {CONVERT_HINT!r}
_MESH_PIE_HINT = {MESH_PIE_HINT!r}
_CATMULL_CLARK_HINT = {CATMULL_CLARK_HINT!r}
_XATLAS_MAX_TRIANGLES = {XATLAS_MAX_TRIANGLES!r}
_MAX_RESULT_TRIANGLES = {MAX_RESULT_TRIANGLES!r}
_DEFAULT_ASSET_FOLDER = {DEFAULT_ASSET_FOLDER!r}
_EXPORT_HINT = {EXPORT_HINT!r}
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

def static_component(actor):
    root = actor.root_component
    if isinstance(root, unreal.StaticMeshComponent):
        return root
    found = list(actor.get_components_by_class(unreal.StaticMeshComponent))
    if len(found) == 1:
        return found[0]
    if found:
        refuse("AMBIGUOUS_STATIC_MESH", actor.get_actor_label() + " has " + str(len(found)) + " StaticMeshComponents "
               "and none is its root, so the mesh to read is ambiguous", actor_path=actor.get_path_name())
    return None

def load_static_mesh(path):
    asset = unreal.EditorAssetLibrary.load_asset(path)
    if asset is None or not isinstance(asset, unreal.StaticMesh):
        if pie_active():
            # The editor asset library loads nothing while PIE runs.
            raise MeshError(_PIE_ACTIVE, _PIE_ERROR.format(action="load assets"), hint=_MESH_PIE_HINT)
        raise MeshError("NotFoundError", "StaticMesh asset not found: " + path)
    return asset

def static_mesh_copy(asset):
    mesh = unreal.DynamicMesh()
    result = gs("copy_mesh_from_static_mesh", unreal.GeometryScript_AssetUtils.copy_mesh_from_static_mesh, asset, mesh,
                unreal.GeometryScriptCopyMeshFromAssetOptions(), unreal.GeometryScriptMeshReadLOD())
    if isinstance(result, tuple) and result[-1] != unreal.GeometryScriptOutcomePins.SUCCESS:
        raise MeshError("EngineError", "GeometryScript could not read the StaticMesh " + asset.get_path_name()
                        + "; nothing was changed", "GEOMETRY_SCRIPT_FAILED")
    return mesh

def read_source(path):
    # A DynamicMeshActor's mesh, or a StaticMeshActor's or StaticMesh asset's first LOD as a
    # transient DynamicMesh, with what the new-output tools need around it.
    try:
        actor = actor_at(path)
    except SimulNotFound as missing:
        kind = asset_class(path)
        if kind == "StaticMesh":
            asset = load_static_mesh(path)
            return {"mesh": static_mesh_copy(asset), "actor": None, "component": None, "asset": asset,
                    "transform": unreal.Transform(), "label": asset.get_name(), "kind": "static_mesh_asset"}
        if kind is not None:
            refuse("NOT_A_MESH", path + " is a " + kind + " asset; give a DynamicMeshActor, a StaticMeshActor or "
                   "a StaticMesh asset")
        raise MeshError("NotFoundError", str(missing))
    root = actor.root_component
    dynamic = [root] if isinstance(root, unreal.DynamicMeshComponent) else list(
        actor.get_components_by_class(unreal.DynamicMeshComponent))
    if not dynamic:
        component = static_component(actor)
        if component is not None:
            asset = component.get_editor_property("static_mesh")
            if asset is None:
                refuse("NO_MESH", actor.get_actor_label() + "'s StaticMeshComponent has no StaticMesh",
                       actor_path=actor.get_path_name())
            return {"mesh": static_mesh_copy(asset), "actor": actor, "component": None, "asset": asset,
                    "transform": actor.get_actor_transform(), "label": actor.get_actor_label(),
                    "kind": "static_mesh_actor"}
    component = mesh_component(actor)
    return {"mesh": component.get_dynamic_mesh(), "actor": actor, "component": component, "asset": None,
            "transform": actor.get_actor_transform(), "label": actor.get_actor_label(), "kind": "dynamic_mesh_actor"}

def require_game_asset(asset, what):
    path = asset.get_path_name()
    if not path.startswith("/Game/"):
        raise MeshError("SandboxError", "Refusing to " + what + " " + path + ": simul writes only assets under "
                        "/Game, never engine or plugin content; nothing was changed", "OUTSIDE_GAME_CONTENT",
                        asset_path=path)

def save_asset(asset):
    path = asset.get_path_name().partition(".")[0]
    saved = bool(unreal.EditorAssetLibrary.save_asset(path, False))
    if not saved:
        raise MeshError("EngineError", "Unreal did not save " + path + "; the change is in the editor, unsaved",
                        "SAVE_FAILED", asset_path=path)
    relative = path[len("/Game/"):] + ".uasset"
    return unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_content_dir()) + relative

def static_collision_count(asset):
    collision = unreal.GeometryScript_Collision.get_simple_collision_from_static_mesh(asset)
    return unreal.GeometryScript_Collision.get_simple_collision_shape_count(collision)

def set_static_collision(asset, collision):
    options = unreal.GeometryScriptSetSimpleCollisionOptions()
    options.emit_transaction = True
    gs("set_simple_collision_of_static_mesh", unreal.GeometryScript_Collision.set_simple_collision_of_static_mesh,
       collision, asset, options, unreal.GeometryScriptSetStaticMeshCollisionOptions())

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

def closed_volume(mesh):
    return volume_of(mesh) if unreal.GeometryScript_MeshQueries.get_is_closed_mesh(mesh) else None

def bounds_of(mesh):
    box = unreal.GeometryScript_MeshQueries.get_mesh_bounding_box(mesh)
    return [box.min.x, box.min.y, box.min.z], [box.max.x, box.max.y, box.max.z]

def working_copy(mesh):
    copy = unreal.DynamicMesh()
    gs("copy", unreal.GeometryScript_MeshDecomposition.copy_mesh_to_mesh, mesh, copy)
    return copy

def vector(values):
    return unreal.Vector(*values)

def triangle_selection():
    return unreal.GeometryScriptMeshSelectionType.TRIANGLES

def selected(mesh, selection):
    return unreal.GeometryScript_MeshSelection.get_mesh_unique_selection_info(mesh, selection)[1]

def commit(actor, component, mesh, description):
    with unreal.ScopedEditorTransaction(description):
        actor.modify()
        component.modify()
        component.set_dynamic_mesh(mesh)

def place_mesh_actor(label, transform, mesh):
    # Spawns inside the caller's transaction; the caller destroys it if a later step fails.
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = actors.spawn_actor_from_class(unreal.DynamicMeshActor, transform.translation,
                                          transform.rotation.rotator())
    if actor is None:
        raise MeshError("EngineError", "Unreal did not spawn a DynamicMeshActor; nothing was changed")
    actor.set_actor_transform(transform, False, False)
    actor.set_actor_label(label)
    actor.dynamic_mesh_component.set_dynamic_mesh(mesh)
    return actor

def in_transaction(description, change):
    # Runs change() in one transaction; actors it records in `placed` are destroyed and the
    # transaction cancelled if it raises, so a failure leaves the level as it was.
    placed = []
    with unreal.ScopedEditorTransaction(description) as transaction:
        try:
            result = change(placed)
        except Exception:
            actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
            for actor in placed:
                actors.destroy_actor(actor)
            transaction.cancel()
            raise
    return result

def spawn_mesh_actor(label, transform, mesh, description):
    def change(placed):
        actor = place_mesh_actor(label, transform, mesh)
        placed.append(actor)
        return actor
    return in_transaction(description, change)

def set_simple_collision(component, collision):
    # With emit_transaction the call also records the component's BodySetup, whose shapes are the
    # ones Unreal collides with and reports; its transaction joins the caller's, so one Undo still
    # reverts the whole edit. Without it the BodySetup kept the new shapes after an Undo (UE 5.7.4).
    options = unreal.GeometryScriptSetSimpleCollisionOptions()
    options.emit_transaction = True
    gs("set_simple_collision_of_dynamic_mesh_component",
       unreal.GeometryScript_Collision.set_simple_collision_of_dynamic_mesh_component, collision, component, options)

def collision_shapes(component):
    collision = unreal.GeometryScript_Collision.get_simple_collision_from_component(component)
    return unreal.GeometryScript_Collision.get_simple_collision_shape_count(collision)

def convex_collision(mesh, hulls, faces, error_tolerance=0.0, min_part_thickness=0.1):
    # Convex hulls as simple collision, and a triangle mesh of the same shapes. The detection of
    # spheres, boxes and capsules is off: with it on, a mesh detected as one gets that shape instead
    # of hulls, which ComputeMeshConvexDecomposition reported as "Hull Computation Failed" and
    # replaced with a 100 cm cube (UE 5.7.4).
    options = unreal.GeometryScriptCollisionFromMeshOptions()
    options.method = unreal.GeometryScriptCollisionGenerationMethod.CONVEX_HULLS
    options.auto_detect_spheres = False
    options.auto_detect_boxes = False
    options.auto_detect_capsules = False
    options.max_convex_hulls_per_mesh = hulls
    options.simplify_hulls = faces > 0
    if faces > 0:
        options.convex_hull_target_face_count = faces
    options.convex_decomposition_error_tolerance = error_tolerance
    options.convex_decomposition_min_part_thickness = min_part_thickness
    options.emit_transaction = False
    collision = gs("generate_collision_from_mesh", unreal.GeometryScript_Collision.generate_collision_from_mesh,
                   mesh, options)
    count = unreal.GeometryScript_Collision.get_simple_collision_shape_count(collision)
    if count == 0:
        raise MeshError("EngineError", "GeometryScript computed no convex shape for the mesh; nothing was "
                        "changed", "EMPTY_RESULT")
    shapes = unreal.DynamicMesh()
    gs("append_simple_collision_shapes", unreal.GeometryScript_Primitives.append_simple_collision_shapes,
       shapes, unreal.GeometryScriptPrimitiveOptions(), unreal.Transform(), collision,
       unreal.GeometryScriptSimpleCollisionTriangulationOptions())
    return collision, shapes, count

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

# Expects args: primitive_type, dimensions (every dimension of the type, cm),
# segments, location, label.
GENERATE_PRIMITIVE = """
def append_primitive(mesh):
    prims = unreal.GeometryScript_Primitives
    options = unreal.GeometryScriptPrimitiveOptions()
    # Quad polygroups, so Catmull-Clark can subdivide the result.
    options.polygroup_mode = unreal.GeometryScriptPrimitivePolygroupMode.PER_QUAD
    xf = unreal.Transform()
    origin = unreal.GeometryScriptPrimitiveOriginMode.CENTER
    d = args["dimensions"]
    seg = args["segments"]
    kind = args["primitive_type"]
    if kind == "box":
        gs("append_box", prims.append_box, mesh, options, xf, d["width"], d["depth"], d["height"], 0, 0, 0, origin)
    elif kind == "sphere":
        gs("append_sphere_lat_long", prims.append_sphere_lat_long, mesh, options, xf, d["radius"], seg, seg, origin)
    elif kind == "cylinder":
        gs("append_cylinder", prims.append_cylinder, mesh, options, xf, d["radius"], d["height"], seg, 0, True,
           origin)
    elif kind == "cone":
        gs("append_cone", prims.append_cone, mesh, options, xf, d["base_radius"], d["top_radius"], d["height"],
           seg, 0, True, origin)
    elif kind == "capsule":
        gs("append_capsule", prims.append_capsule, mesh, options, xf, d["radius"], d["length"],
           max(seg // 4, 2), seg, 0, origin)
    else:
        gs("append_torus", prims.append_torus, mesh, options, xf, unreal.GeometryScriptRevolveOptions(),
           d["major_radius"], d["minor_radius"], seg, max(seg // 2, 3), origin)

def main():
    mesh = unreal.DynamicMesh()
    append_primitive(mesh)
    transform = unreal.Transform()
    transform.translation = vector(args["location"])
    actor = spawn_mesh_actor(args["label"], transform, mesh, "simul: generate mesh " + args["primitive_type"])
    low, high = bounds_of(mesh)
    return {"actor_path": actor.get_path_name(), "primitive_type": args["primitive_type"],
            "dimensions": args["dimensions"], "segments": args["segments"],
            "triangle_count": triangle_count(mesh), "vertex_count": vertex_count(mesh),
            "bounds_min": low, "bounds_max": high}

run(main, "spawn actors")
"""

# Expects args: target, tool, operation (canonical), requested_operation.
MESH_BOOLEAN = f"""
_OPERATIONS = {BOOLEAN_OPERATIONS!r}
""" + """
def main():
    target = mesh_actor(args["target"])
    tool = mesh_actor(args["tool"])
    if target.get_path_name() == tool.get_path_name():
        refuse("SAME_ACTOR", "The target and the tool are the same actor; give two DynamicMeshActors")
    component = mesh_component(target)
    work = working_copy(component.get_dynamic_mesh())
    before = triangle_count(work)
    before_volume = closed_volume(work)
    options = unreal.GeometryScriptMeshBooleanOptions()
    # Both meshes at their world transforms; the result in the target's local space.
    options.output_transform_space = unreal.GeometryScriptBooleanOutputSpace.TARGET_TRANSFORM_SPACE
    gs("apply_mesh_boolean", unreal.GeometryScript_MeshBooleans.apply_mesh_boolean, work,
       target.get_actor_transform(), mesh_component(tool).get_dynamic_mesh(), tool.get_actor_transform(),
       getattr(unreal.GeometryScriptBooleanOperation, _OPERATIONS[args["operation"]]), options)
    commit(target, component, work, "simul: mesh boolean " + args["operation"])
    return {"target_mesh_path": target.get_path_name(), "tool_mesh_path": tool.get_path_name(),
            "operation": args["operation"], "requested_operation": args["requested_operation"],
            "previous_triangle_count": before, "result_triangle_count": triangle_count(work),
            "result_vertex_count": vertex_count(work), "previous_volume": before_volume,
            "result_volume": closed_volume(work), "result_components": first(
                unreal.GeometryScript_MeshQueries.get_num_connected_components(work))}

run(main, "edit meshes")
"""

# Shared by the hull and the decomposition: put the shapes where `output` says. A
# StaticMesh source's collision goes on its asset (every actor using it), which
# must be under /Game; `save` then saves that asset.
_PLACE_CONVEX = """
def check_collision_target(source, output):
    if output in ("collision", "both") and source["asset"] is not None:
        require_game_asset(source["asset"], "set the simple collision of")

def place_convex(source, collision, shapes, output, suffix, description, save):
    def change(placed):
        actor = None
        targets = []
        if output in ("actor", "both"):
            actor = place_mesh_actor(source["label"] + suffix, source["transform"], shapes)
            placed.append(actor)
            set_simple_collision(actor.dynamic_mesh_component, collision)
            targets.append(("actor", actor))
        if output in ("collision", "both"):
            if source["component"] is not None:
                source["actor"].modify()
                source["component"].modify()
                set_simple_collision(source["component"], collision)
                targets.append(("actor", source["actor"]))
            else:
                source["asset"].modify()
                set_static_collision(source["asset"], collision)
                targets.append(("asset", source["asset"]))
        return actor, targets
    actor, targets = in_transaction(description, change)
    collided = []
    for kind, target in targets:
        if kind == "actor":
            collided.append({"actor_path": target.get_path_name(), "asset_path": None,
                             "collision_shape_count": collision_shapes(mesh_component(target)), "saved_file": None})
        else:
            collided.append({"actor_path": None, "asset_path": target.get_path_name(),
                             "collision_shape_count": static_collision_count(target),
                             "saved_file": save_asset(target) if save else None})
    return actor, collided
"""

# Expects args: mesh, max_faces, output, save.
CONVEX_HULL = _PLACE_CONVEX + """
def main():
    source = read_source(args["mesh"])
    check_collision_target(source, args["output"])
    mesh = source["mesh"]
    collision, hull, count = convex_collision(mesh, 1, args["max_faces"])
    source_volume = closed_volume(mesh)
    hull_volume = volume_of(hull)
    actor, collided = place_convex(source, collision, hull, args["output"], "_Hull",
                                   "simul: convex hull of " + source["label"], args["save"])
    return {"mesh_path": args["mesh"], "source_kind": source["kind"], "output": args["output"],
            "hull_actor_path": actor.get_path_name() if actor is not None else None,
            "hull_vertex_count": vertex_count(hull), "hull_triangle_count": triangle_count(hull),
            "max_faces": args["max_faces"], "source_triangle_count": triangle_count(mesh),
            "hull_volume": hull_volume, "source_volume": source_volume,
            "volume_ratio": hull_volume / source_volume if source_volume else None, "collision": collided}

run(main, "spawn actors" if args["output"] != "collision" else "edit meshes")
"""

# Expects args: mesh, max_hulls, max_vertices_per_hull, error_tolerance,
# min_part_thickness, output, save.
CONVEX_DECOMPOSITION = _PLACE_CONVEX + """
def hull_info(index, part):
    low, high = bounds_of(part)
    return {"index": index, "vertex_count": vertex_count(part), "triangle_count": triangle_count(part),
            "volume": volume_of(part), "bounds_min": low, "bounds_max": high}

def main():
    source = read_source(args["mesh"])
    check_collision_target(source, args["output"])
    mesh = source["mesh"]
    # A closed convex hull of V vertices has 2V - 4 triangles.
    faces = 2 * args["max_vertices_per_hull"] - 4
    collision, shapes, count = convex_collision(mesh, args["max_hulls"], faces, args["error_tolerance"],
                                                args["min_part_thickness"])
    parts = gs("split_mesh_by_components", unreal.GeometryScript_MeshDecomposition.split_mesh_by_components,
               shapes, None)[1]
    hulls = [hull_info(index, part) for index, part in enumerate(parts)]
    source_volume = closed_volume(mesh)
    total_volume = sum(hull["volume"] for hull in hulls)
    actor, collided = place_convex(source, collision, shapes, args["output"], "_Decomp",
                                   "simul: convex decomposition of " + source["label"], args["save"])
    return {"mesh_path": args["mesh"], "source_kind": source["kind"], "output": args["output"],
            "decomp_actor_path": actor.get_path_name() if actor is not None else None,
            "hull_count": len(hulls), "hulls": hulls, "max_hulls": args["max_hulls"],
            "max_vertices_per_hull": args["max_vertices_per_hull"],
            "largest_hull_vertex_count": max(hull["vertex_count"] for hull in hulls),
            "total_vertices": vertex_count(shapes), "total_triangles": triangle_count(shapes),
            "hulls_volume": total_volume, "source_volume": source_volume,
            "volume_ratio": total_volume / source_volume if source_volume else None, "collision": collided}

run(main, "spawn actors" if args["output"] != "collision" else "edit meshes")
"""

# Expects args: mesh, operation, faces (a face selection or None), edges (an
# edge selection or None), distance, offset, subdivisions. Selections are in
# the mesh's local space.
EDIT_TOPOLOGY = """
def face_selection(work, spec):
    selections = unreal.GeometryScript_MeshSelection
    if spec["kind"] == "all":
        selection = selections.create_select_all_mesh_selection(work, triangle_selection())[1]
    elif spec["kind"] == "normal":
        selection = selections.select_mesh_elements_by_normal_angle(
            work, vector(spec["normal"]), spec["angle"], triangle_selection(), False, 3)[1]
    else:
        box = unreal.Box()
        box.min = vector(spec["min"])
        box.max = vector(spec["max"])
        box.is_valid = True
        selection = selections.select_mesh_elements_in_box(work, box, triangle_selection(), False, 3)[1]
    count = selected(work, selection)
    if count == 0:
        refuse("EMPTY_SELECTION", "face_selection " + repr(spec["text"]) + " selects no triangle of the mesh "
               "(selections are in the mesh's local space); nothing was changed")
    return selection, count

def apply(work):
    modeling = unreal.GeometryScript_MeshModeling
    operation = args["operation"]
    faces = edges = None
    if operation == "extrude_faces":
        selection, faces = face_selection(work, args["faces"])
        options = unreal.GeometryScriptMeshLinearExtrudeOptions()
        options.distance = args["distance"]
        options.direction_mode = unreal.GeometryScriptLinearExtrudeDirection.AVERAGE_FACE_NORMAL
        gs("apply_mesh_linear_extrude_faces", modeling.apply_mesh_linear_extrude_faces, work, options, selection)
    elif operation == "inset_faces":
        selection, faces = face_selection(work, args["faces"])
        options = unreal.GeometryScriptMeshInsetOutsetFacesOptions()
        options.distance = args["offset"]
        # Each polygroup (a primitive's face) of the selection is inset on its own.
        options.area_mode = unreal.GeometryScriptPolyOperationArea.PER_POLYGROUP
        gs("apply_mesh_inset_outset_faces", modeling.apply_mesh_inset_outset_faces, work, options, selection)
    else:
        options = unreal.GeometryScriptMeshBevelSelectionOptions()
        options.bevel_distance = args["distance"]
        options.subdivisions = args["subdivisions"]
        modes = unreal.GeometryScriptMeshBevelSelectionMode
        selections = unreal.GeometryScript_MeshSelection
        if args["faces"] is not None:
            selection, faces = face_selection(work, args["faces"])
            edges = selected(work, selections.select_selection_boundary_edges(work, selection, False)[1])
            mode = modes.TRIANGLE_AREA
        elif args["edges"]["kind"] == "sharp":
            selection = selections.select_mesh_sharp_edges(work, args["edges"]["angle"])[1]
            edges = selected(work, selection)
            if edges == 0:
                refuse("EMPTY_SELECTION", "No edge of the mesh is sharper than " + str(args["edges"]["angle"])
                       + " degrees; nothing was changed")
            mode = modes.SELECTED_EDGES
        else:
            selection = selections.create_select_all_mesh_selection(work, triangle_selection())[1]
            mode = modes.ALL_POLYGROUP_EDGES
        gs("apply_mesh_bevel_selection", modeling.apply_mesh_bevel_selection, work, selection, mode, options)
    return faces, edges

def main():
    actor = mesh_actor(args["mesh"])
    component = mesh_component(actor)
    work = working_copy(component.get_dynamic_mesh())
    before = triangle_count(work)
    faces, edges = apply(work)
    commit(actor, component, work, "simul: mesh " + args["operation"])
    return {"mesh_path": actor.get_path_name(), "operation": args["operation"],
            "face_selection": args["faces"]["text"] if args["faces"] else None,
            "edge_selection": args["edges"]["text"] if args["edges"] else None,
            "faces_affected": faces, "edges_affected": edges,
            "distance": args["distance"] if args["operation"] != "inset_faces" else None,
            "offset": args["offset"] if args["operation"] == "inset_faces" else None,
            "subdivisions": args["subdivisions"] if args["operation"] == "bevel_edges" else None,
            "previous_triangle_count": before, "result_triangle_count": triangle_count(work),
            "result_vertex_count": vertex_count(work), "result_volume": closed_volume(work)}

run(main, "edit meshes")
"""

# Expects args: mesh, level, scheme.
SUBDIVIDE = """
def predicted(triangles):
    if args["scheme"] == "uniform":
        return triangles * (args["level"] + 1) ** 2
    return triangles * 4 ** args["level"]

def main():
    actor = mesh_actor(args["mesh"])
    component = mesh_component(actor)
    mesh = component.get_dynamic_mesh()
    before = triangle_count(mesh)
    if predicted(before) > _MAX_RESULT_TRIANGLES:
        refuse("RESULT_TOO_LARGE", "Subdividing " + str(before) + " triangles with " + args["scheme"] + " level "
               + str(args["level"]) + " would make about " + str(predicted(before)) + " triangles, more than "
               + str(_MAX_RESULT_TRIANGLES) + "; use a lower level or simplify first",
               predicted_triangle_count=predicted(before), max_triangles=_MAX_RESULT_TRIANGLES)
    work = working_copy(mesh)
    if args["scheme"] == "catmull_clark":
        try:
            gs("apply_polygroup_catmull_clark_sub_d",
               unreal.GeometryScript_OpenSubdiv.apply_polygroup_catmull_clark_sub_d,
               work, args["level"], unreal.GeometryScriptGroupLayer())
        except MeshError as failure:
            failure.details["hint"] = _CATMULL_CLARK_HINT
            raise
    elif args["scheme"] == "loop":
        gs("apply_triangle_loop_sub_d", unreal.GeometryScript_OpenSubdiv.apply_triangle_loop_sub_d, work,
           args["level"])
    else:
        gs("apply_uniform_tessellation", unreal.GeometryScript_MeshSubdivide.apply_uniform_tessellation, work,
           args["level"])
    commit(actor, component, work, "simul: subdivide mesh (" + args["scheme"] + ")")
    return {"mesh_path": actor.get_path_name(), "level": args["level"], "scheme": args["scheme"],
            "previous_triangle_count": before, "result_triangle_count": triangle_count(work),
            "result_vertex_count": vertex_count(work)}

run(main, "edit meshes")
"""

# Expects args: mesh, target_triangle_count, target_percentage, max_error
# (at most one set).
SIMPLIFY = """
def main():
    actor = mesh_actor(args["mesh"])
    component = mesh_component(actor)
    mesh = component.get_dynamic_mesh()
    before = triangle_count(mesh)
    simplifier = unreal.GeometryScript_MeshSimplification
    if args["max_error"] is not None:
        target, method = None, "tolerance"
    elif args["target_triangle_count"] is not None:
        target, method = args["target_triangle_count"], "triangle_count"
    else:
        fraction = args["target_percentage"] if args["target_percentage"] is not None else 0.5
        target, method = max(4, int(round(before * fraction))), "triangle_count"
    if target is not None and target >= before:
        refuse("NOTHING_TO_SIMPLIFY", actor.get_actor_label() + " has " + str(before) + " triangles, no more than "
               "the target of " + str(target) + "; nothing was changed", triangle_count=before,
               target_triangle_count=target)
    work = working_copy(mesh)
    if method == "tolerance":
        gs("apply_simplify_to_tolerance", simplifier.apply_simplify_to_tolerance, work, args["max_error"],
           unreal.GeometryScriptSimplifyMeshOptions())
    else:
        gs("apply_simplify_to_triangle_count", simplifier.apply_simplify_to_triangle_count, work, target,
           unreal.GeometryScriptSimplifyMeshOptions())
    after = triangle_count(work)
    commit(actor, component, work, "simul: simplify mesh")
    return {"mesh_path": actor.get_path_name(), "method": method, "target_triangle_count": target,
            "tolerance": args["max_error"], "original_triangles": before, "result_triangles": after,
            "result_vertex_count": vertex_count(work),
            "reduction_ratio": round(1.0 - after / before, 4) if before > 0 else 0.0}

run(main, "edit meshes")
"""

# Expects args: mesh, origin, normal, space ("world" or "local"), fill_holes,
# keep_both_sides.
PLANE_CUT = """
def local_plane(actor):
    origin, normal = vector(args["origin"]), vector(args["normal"])
    if args["space"] == "local":
        return origin, normal
    # A world plane through o with normal n is, in the mesh's space (world = R(S p) + t), the plane
    # through T^-1(o) with normal S (R^-1 n): normals take the scale, not its inverse.
    transform = actor.get_actor_transform()
    local_origin = transform.inverse_transform_location(origin)
    unrotated = transform.rotation.unrotate_vector(normal)
    scale = transform.scale3d
    x, y, z = unrotated.x * scale.x, unrotated.y * scale.y, unrotated.z * scale.z
    length = math.sqrt(x * x + y * y + z * z)
    return local_origin, unreal.Vector(x / length, y / length, z / length)

def cut(mesh, origin, normal, fill, flip):
    work = working_copy(mesh)
    frame = unreal.Transform()
    frame.translation = origin
    frame.rotation = unreal.MathLibrary.make_rot_from_z(normal).quaternion()
    options = unreal.GeometryScriptMeshPlaneCutOptions()
    options.fill_holes = fill
    options.flip_cut_side = flip
    gs("apply_mesh_plane_cut", unreal.GeometryScript_MeshBooleans.apply_mesh_plane_cut, work, frame, options)
    return work

def main():
    actor = mesh_actor(args["mesh"])
    component = mesh_component(actor)
    mesh = component.get_dynamic_mesh()
    origin, normal = local_plane(actor)
    before = triangle_count(mesh)
    # GeometryScript keeps the side the normal points away from; flip_cut_side keeps the other.
    kept = cut(mesh, origin, normal, args["fill_holes"], False)
    other = cut(mesh, origin, normal, args["fill_holes"], True)
    if triangle_count(kept) == 0 or triangle_count(other) == 0:
        side = "in front of" if triangle_count(kept) == 0 else "behind"
        refuse("PLANE_MISSES_MESH", "The plane does not cross " + actor.get_actor_label() + ": all of it is "
               + side + " the plane (the side the normal points to is removed); nothing was changed",
               plane_origin_local=[origin.x, origin.y, origin.z], plane_normal_local=[normal.x, normal.y, normal.z])
    filled = 0
    if args["fill_holes"]:
        filled = triangle_count(kept) - triangle_count(cut(mesh, origin, normal, False, False))

    def change(placed):
        actor.modify()
        component.modify()
        component.set_dynamic_mesh(kept)
        if not args["keep_both_sides"]:
            return None
        piece = place_mesh_actor(actor.get_actor_label() + "_Cut", actor.get_actor_transform(), other)
        placed.append(piece)
        return piece

    piece = in_transaction("simul: cut mesh", change)
    pieces = [actor.get_path_name()] + ([piece.get_path_name()] if piece is not None else [])
    counts = [triangle_count(kept)] + ([triangle_count(other)] if piece is not None else [])
    return {"mesh_path": actor.get_path_name(), "space": args["space"],
            "plane_origin_local": [origin.x, origin.y, origin.z],
            "plane_normal_local": [normal.x, normal.y, normal.z], "pieces": pieces,
            "piece_triangle_counts": counts, "removed_triangle_count": None if piece else triangle_count(other),
            "cut_faces_added": filled, "previous_triangle_count": before,
            "result_triangle_count": triangle_count(kept), "result_vertex_count": vertex_count(kept)}

run(main, "edit meshes")
"""

# Expects args: mesh, checks (names from VALIDATION_CHECKS).
VALIDATE = """
def check(passed, count, detail):
    return {"passed": bool(passed), "count": count, "detail": detail}

def run_check(name, mesh):
    queries = unreal.GeometryScript_MeshQueries
    if name == "closed":
        edges = first(queries.get_num_open_border_edges(mesh))
        loops = first(queries.get_num_open_border_loops(mesh))
        return check(edges == 0, edges, str(edges) + " open border edges in " + str(loops) + " loops"
                     if edges else "no open border edges")
    if name == "manifold":
        # Splitting bowties adds one vertex per extra fan at a non-manifold vertex.
        copy = working_copy(mesh)
        before = vertex_count(copy)
        gs("split_mesh_bowties", unreal.GeometryScript_MeshRepair.split_mesh_bowties, copy, True, False)
        bowties = vertex_count(copy) - before
        return check(bowties == 0, bowties, str(bowties) + " non-manifold (bowtie) vertices" if bowties
                     else "no non-manifold vertices")
    if name == "degenerate_triangles":
        copy = working_copy(mesh)
        options = unreal.GeometryScriptDegenerateTriangleOptions()
        options.mode = unreal.GeometryScriptRepairMeshMode.DELETE_ONLY
        before = triangle_count(copy)
        gs("repair_mesh_degenerate_geometry", unreal.GeometryScript_MeshRepair.repair_mesh_degenerate_geometry,
           copy, options)
        degenerate = before - triangle_count(copy)
        return check(degenerate == 0, degenerate, str(degenerate) + " triangles with an area under "
                     + str(options.min_triangle_area) + " cm2 or an edge under " + str(options.min_edge_length)
                     + " cm" if degenerate else "no degenerate triangles")
    if name == "orientation":
        if not queries.get_is_closed_mesh(mesh):
            return check(True, None, "not checked: the mesh is not closed, so it encloses no volume")
        volume = volume_of(mesh)
        return check(volume > 0, None, "encloses " + str(round(volume, 3)) + " cm3" + (
            "" if volume > 0 else ": the triangles face inward (inside out)"))
    if name == "compact":
        triangles, vertices = bool(queries.get_has_triangle_id_gaps(mesh)), bool(queries.get_has_vertex_id_gaps(mesh))
        return check(not (triangles or vertices), None, "triangle or vertex ID gaps (not compact)"
                     if triangles or vertices else "no ID gaps")
    if name == "normals":
        has = bool(queries.get_has_triangle_normals(mesh))
        return check(has, None, "has a normals attribute" if has else "no normals attribute")
    channels = queries.get_num_uv_sets(mesh)
    if channels == 0:
        return check(False, 0, "no UV channel")
    info = unreal.GeometryScript_UVs.get_mesh_uv_size_info(mesh, 0, unreal.GeometryScriptMeshSelection())
    unset = bool(info[-1])
    return check(not unset, channels, str(channels) + " UV channels" + (
        "; channel 0 has triangles without UVs" if unset else ""))

def main():
    queries = unreal.GeometryScript_MeshQueries
    source = read_source(args["mesh"])
    mesh = source["mesh"]
    checks = {name: run_check(name, mesh) for name in args["checks"]}
    low, high = bounds_of(mesh)
    area, volume = queries.get_mesh_volume_area(mesh)
    return {"mesh_path": args["mesh"], "source_kind": source["kind"],
            "is_valid": all(c["passed"] for c in checks.values()),
            "checks": checks, "issues": [name + ": " + c["detail"] for name, c in checks.items() if not c["passed"]],
            "triangle_count": triangle_count(mesh), "vertex_count": vertex_count(mesh),
            "connected_components": first(queries.get_num_connected_components(mesh)),
            "uv_channels": queries.get_num_uv_sets(mesh), "bounds_min": low, "bounds_max": high,
            "surface_area": float(area), "volume": float(volume) if queries.get_is_closed_mesh(mesh) else None}

run(main)
"""

# Expects args: mesh, target_format, location (dynamic_mesh from an asset),
# asset_path, overwrite, save, replace_actor, collision, nanite.
CONVERT = """
def material_list(asset):
    result = unreal.GeometryScript_AssetUtils.get_section_material_list_from_static_mesh(
        asset, unreal.GeometryScriptMeshReadLOD())
    return list(result[0])

def to_dynamic_mesh():
    path = args["mesh"]
    try:
        actor = actor_at(path)
    except SimulNotFound as missing:
        actor = None
        kind = asset_class(path)
        if kind is None:
            raise MeshError("NotFoundError", str(missing))
        if kind != "StaticMesh":
            refuse("NOT_A_STATIC_MESH", path + " is a " + kind + " asset, not a StaticMesh")
    if actor is not None:
        component = None if actor.get_components_by_class(unreal.DynamicMeshComponent) else static_component(actor)
        if component is None:
            refuse("NOT_A_STATIC_MESH", actor.get_actor_label() + " (" + actor.get_class().get_name() + ") renders no "
                   "StaticMesh; give a StaticMeshActor or a StaticMesh asset")
        asset = component.get_editor_property("static_mesh")
        if asset is None:
            refuse("NO_MESH", actor.get_actor_label() + "'s StaticMeshComponent has no StaticMesh")
        transform, label, kind = actor.get_actor_transform(), actor.get_actor_label(), "static_mesh_actor"
    else:
        if args["replace_actor"]:
            refuse("NOTHING_TO_REPLACE", "replace_actor needs a StaticMeshActor; " + path + " is an asset")
        asset = load_static_mesh(path)
        transform = unreal.Transform()
        transform.translation = vector(args["location"])
        label, kind = asset.get_name(), "static_mesh_asset"
    mesh = static_mesh_copy(asset)
    materials = material_list(asset)

    def change(placed):
        placed_actor = place_mesh_actor(label + ("" if args["replace_actor"] else "_Dynamic"), transform, mesh)
        placed.append(placed_actor)
        unreal.GeometryScript_SceneUtils.set_component_material_list(placed_actor.dynamic_mesh_component, materials)
        if args["replace_actor"]:
            unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor)
        return placed_actor

    placed_actor = in_transaction("simul: convert " + label + " to a DynamicMeshActor", change)
    return {"source_path": path, "source_format": kind, "result_path": placed_actor.get_path_name(),
            "target_format": "dynamic_mesh", "placed_actor_path": placed_actor.get_path_name(),
            "removed_actor_path": path if args["replace_actor"] else None, "asset": None, "hint": None,
            "triangle_count": triangle_count(mesh), "vertex_count": vertex_count(mesh)}

def nanite_settings(settings):
    settings.enabled = True
    return settings

def asset_collision(component, mesh):
    choice = args["collision"]
    if choice == "auto":
        choice = "component" if collision_shapes(component) > 0 else "convex_hull"
    if choice == "component":
        return choice, unreal.GeometryScript_Collision.get_simple_collision_from_component(component)
    if choice == "convex_hull":
        return choice, convex_collision(mesh, 1, 250)[0]
    return choice, unreal.GeometryScriptSimpleCollision()

def to_static_mesh():
    actor = mesh_actor(args["mesh"])
    component = mesh_component(actor)
    mesh = component.get_dynamic_mesh()
    if triangle_count(mesh) == 0:
        refuse("EMPTY_MESH", actor.get_actor_label() + " has no triangles to make a StaticMesh of")
    name = "".join(c if c.isalnum() or c == "_" else "_" for c in actor.get_actor_label()).strip("_") or "Mesh"
    path = args["asset_path"] or (_DEFAULT_ASSET_FOLDER + "/SM_" + name)
    library = unreal.EditorAssetLibrary
    overwritten = bool(library.does_asset_exist(path))
    if overwritten:
        if asset_class(path) != "StaticMesh":
            refuse("ASSET_PATH_TAKEN", path + " is a " + str(asset_class(path)) + " asset, not a StaticMesh; give "
                   "another asset_path", asset_path=path)
        if not args["overwrite"]:
            refuse("ASSET_EXISTS", "A StaticMesh already exists at " + path + "; pass overwrite true to replace its "
                   "mesh (its references are kept) or give another asset_path", asset_path=path)
    collision_kind, collision = asset_collision(component, mesh)
    if overwritten:
        asset = load_static_mesh(path)
        require_game_asset(asset, "overwrite")
        options = unreal.GeometryScriptCopyMeshToAssetOptions()
        if args["nanite"]:
            options.apply_nanite_settings = True
            options.new_nanite_settings = nanite_settings(asset.get_editor_property("nanite_settings"))
        result = gs("copy_mesh_to_static_mesh", unreal.GeometryScript_AssetUtils.copy_mesh_to_static_mesh, mesh, asset,
                    options, unreal.GeometryScriptMeshWriteLOD(), True)
        if isinstance(result, tuple) and result[-1] != unreal.GeometryScriptOutcomePins.SUCCESS:
            raise MeshError("EngineError", "GeometryScript could not write the mesh into " + path,
                            "GEOMETRY_SCRIPT_FAILED")
    else:
        options = unreal.GeometryScriptCreateNewStaticMeshAssetOptions()
        options.enable_collision = True
        if args["nanite"]:
            # Unreal copies nanite_settings onto the new asset, enabled flag included (its default is off).
            options.enable_nanite = True
            options.nanite_settings = nanite_settings(options.nanite_settings)
        result = gs("create_new_static_mesh_asset_from_mesh",
                    unreal.GeometryScript_NewAssetUtils.create_new_static_mesh_asset_from_mesh, mesh, path, options)
        asset = result[0] if isinstance(result, tuple) else result
        if asset is None or (isinstance(result, tuple) and result[-1] != unreal.GeometryScriptOutcomePins.SUCCESS):
            raise MeshError("EngineError", "GeometryScript did not create a StaticMesh at " + path + " (see the "
                            "editor log); nothing was changed", "GEOMETRY_SCRIPT_FAILED", asset_path=path)
    for index, material in enumerate(component.get_materials()):
        if material is not None and index < len(asset.static_materials):
            asset.set_material(index, material)
    set_static_collision(asset, collision)
    saved_file = save_asset(asset) if args["save"] else None
    placed = None
    if args["replace_actor"]:
        def change(placed_actors):
            actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
            transform = actor.get_actor_transform()
            replacement = actors.spawn_actor_from_class(unreal.StaticMeshActor, transform.translation,
                                                        transform.rotation.rotator())
            if replacement is None:
                raise MeshError("EngineError", "Unreal did not spawn the StaticMeshActor; the asset " + path
                                + " was made, the level was not changed")
            placed_actors.append(replacement)
            replacement.set_actor_transform(transform, False, False)
            replacement.set_actor_label(actor.get_actor_label())
            replacement.static_mesh_component.set_static_mesh(asset)
            actors.destroy_actor(actor)
            return replacement
        placed = in_transaction("simul: replace " + actor.get_actor_label() + " with a StaticMeshActor", change)
    return {"source_path": args["mesh"], "source_format": "dynamic_mesh_actor", "result_path": asset.get_path_name(),
            "target_format": "static_mesh", "placed_actor_path": placed.get_path_name() if placed else None,
            "removed_actor_path": args["mesh"] if placed else None,
            "asset": {"asset_path": asset.get_path_name(), "overwritten": overwritten, "saved_file": saved_file,
                      "collision": collision_kind, "collision_shape_count": static_collision_count(asset),
                      "material_slots": len(asset.static_materials),
                      "nanite": bool(asset.get_editor_property("nanite_settings").enabled),
                      "triangle_count": asset.get_num_triangles(0), "vertex_count": asset.get_num_vertices(0)},
            "hint": None if placed else _EXPORT_HINT,
            "triangle_count": triangle_count(mesh), "vertex_count": vertex_count(mesh)}

def main():
    if args["target_format"] == "dynamic_mesh":
        return to_dynamic_mesh()
    return to_static_mesh()

run(main, "spawn actors" if args["target_format"] == "dynamic_mesh" else "save assets")
"""

# Expects args: mesh, target_edge_length, target_triangle_count (at most one
# set; neither keeps the triangle count), passes.
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
    uniform = unreal.GeometryScriptUniformRemeshOptions()
    if args["target_edge_length"] is not None:
        length = args["target_edge_length"]
        area = float(unreal.GeometryScript_MeshQueries.get_mesh_volume_area(mesh)[0])
        estimate = int(4.0 * area / (math.sqrt(3.0) * length * length))
        if estimate > _MAX_RESULT_TRIANGLES:
            refuse("RESULT_TOO_LARGE", "A " + str(length) + " cm edge length over " + str(round(area)) + " cm2 "
                   "makes about " + str(estimate) + " triangles, more than " + str(_MAX_RESULT_TRIANGLES)
                   + "; use a longer edge", predicted_triangle_count=estimate, max_triangles=_MAX_RESULT_TRIANGLES)
        uniform.target_edge_length = length
        uniform.target_type = unreal.GeometryScriptUniformRemeshTargetType.TARGET_EDGE_LENGTH
        target = {"target_type": "edge_length", "target_edge_length": length, "target_triangle_count": None}
    else:
        count = args["target_triangle_count"] if args["target_triangle_count"] is not None else before
        uniform.target_triangle_count = count
        uniform.target_type = unreal.GeometryScriptUniformRemeshTargetType.TRIANGLE_COUNT
        target = {"target_type": "triangle_count", "target_edge_length": None, "target_triangle_count": count}
    options = unreal.GeometryScriptRemeshOptions()
    options.remesh_iterations = args["passes"]
    work = working_copy(mesh)
    gs("apply_uniform_remesh", unreal.GeometryScript_Remeshing.apply_uniform_remesh, work, options, uniform)
    commit(actor, component, work, "simul: remesh mesh")
    return {"mesh_path": actor.get_path_name(), "mode": "uniform", **target, "passes": args["passes"],
            "original_triangles": before, "result_triangles": triangle_count(work),
            "result_vertex_count": vertex_count(work), "average_edge_length": mean_edge_length(work)}

run(main, "edit meshes")
"""

# Expects args: mesh, method ("auto", "xatlas" or "patch_builder"),
# uv_channel, island_padding (texels), texture_resolution.
COMPUTE_UV = """
def packing_resolution():
    # Unreal's packer leaves a one-texel gutter at the resolution it packs for, so a gutter of
    # island_padding texels at texture_resolution is one texel at texture_resolution / padding.
    return int(max(16, min(4096, round(args["texture_resolution"] / args["island_padding"]))))

def unwrap(work, method, channel, resolution):
    uvs = unreal.GeometryScript_UVs
    packing = unreal.GeometryScriptRepackUVsOptions()
    packing.target_image_width = resolution
    packing.optimize_island_rotation = True
    if method == "xatlas":
        gs("auto_generate_x_atlas_mesh_u_vs", uvs.auto_generate_x_atlas_mesh_u_vs, work, channel,
           unreal.GeometryScriptXAtlasOptions())
        gs("repack_mesh_u_vs", uvs.repack_mesh_u_vs, work, channel, packing)
    else:
        options = unreal.GeometryScriptPatchBuilderOptions()
        options.auto_pack = True
        options.packing_options = packing
        gs("auto_generate_patch_builder_mesh_u_vs", uvs.auto_generate_patch_builder_mesh_u_vs, work, channel, options)

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
    if channel >= channels:
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
    resolution = packing_resolution()
    work = working_copy(mesh)
    unwrap(work, method, channel, resolution)
    commit(actor, component, work, "simul: compute mesh UVs")
    return {"mesh_path": actor.get_path_name(), "method": method, "requested_method": args["method"],
            "uv_channel": channel, "island_padding": args["texture_resolution"] / resolution,
            "texture_resolution": args["texture_resolution"], "packing_resolution": resolution,
            "island_count": first(unreal.GeometryScript_MeshQueries.get_num_uv_islands(work, channel)),
            "coverage_ratio": coverage(work, channel),
            "triangle_count": triangle_count(work), "vertex_count": vertex_count(work)}

run(main, "edit meshes")
"""

#: The ``details.code`` of a failed GeometryScript call (``EngineError``).
GEOMETRY_SCRIPT_FAILED = "GEOMETRY_SCRIPT_FAILED"


# ---------------------------------------------------------------------------
# Checks the session makes before it asks the editor
# ---------------------------------------------------------------------------


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


def primitive_dimensions(
    kind: str, given: Dict[str, Any]
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
    """Every dimension of a primitive: the given ones checked, the rest defaulted.

    Args:
        kind: The primitive type (one of ``PRIMITIVE_TYPES``).
        given: The dimensions the caller gave.

    Returns:
        ``(dimensions, None)``, or ``(None, refusal)`` for an unknown name, a
        value that is not a finite number above zero (``top_radius`` may be
        0), or a shape that cannot exist (a cone narrower at the base, a torus
        whose tube is as thick as its radius).
    """
    defaults = PRIMITIVE_DIMENSIONS[kind]
    unknown = sorted(set(given) - set(defaults))
    if unknown:
        return None, refusal(
            "UNKNOWN_DIMENSION",
            f"{kind} takes the dimensions {', '.join(defaults)} (cm), not {', '.join(unknown)}",
            dimensions=list(defaults),
        )
    dimensions: Dict[str, float] = {}
    for name, default in defaults.items():
        value = given.get(name, default)
        number = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan
        minimum_ok = number >= 0 if name == "top_radius" else number > 0
        if not (math.isfinite(number) and minimum_ok):
            return None, refusal(
                "INVALID_DIMENSION",
                f"{kind} {name} must be a number above 0 cm{' (or 0)' if name == 'top_radius' else ''}, not {value!r}",
            )
        dimensions[name] = number
    if kind == "cone" and dimensions["top_radius"] >= dimensions["base_radius"]:
        return None, refusal("INVALID_DIMENSION", "A cone's top_radius must be smaller than its base_radius")
    if kind == "torus" and dimensions["minor_radius"] >= dimensions["major_radius"]:
        return None, refusal("INVALID_DIMENSION", "A torus's minor_radius must be smaller than its major_radius")
    return dimensions, None


def _numbers(text: str, count: Tuple[int, ...]) -> Optional[List[float]]:
    try:
        values = [float(part) for part in text.split(",")]
    except ValueError:
        return None
    return values if len(values) in count and all(math.isfinite(v) for v in values) else None


#: How face and edge selections are written, for the refusal of a bad one.
FACE_SELECTION_SYNTAX = (
    "'all'; 'normal:x,y,z' or 'normal:x,y,z,max_angle_deg' (triangles facing that way, within 15 degrees "
    "by default); or 'box:min_x,min_y,min_z,max_x,max_y,max_z' (triangles inside the box), in the "
    "mesh's local space"
)
EDGE_SELECTION_SYNTAX = (
    "'polygroup' (every polygroup edge, a primitive's face edges) or 'sharp' / 'sharp:min_angle_deg' "
    "(edges whose faces meet at 20 degrees or more by default)"
)


def face_selection(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Parse a face selection (``FACE_SELECTION_SYNTAX``).

    Returns:
        ``(selection, None)`` or ``(None, refusal)``.
    """
    kind, _, rest = text.strip().lower().partition(":")
    if kind == "all" and not rest:
        return {"kind": "all", "text": text}, None
    if kind == "normal":
        values = _numbers(rest, (3, 4))
        if values is not None and math.hypot(*values[:3]) > 0 and (len(values) == 3 or 0 < values[3] <= 180):
            return {"kind": "normal", "normal": values[:3], "angle": values[3] if len(values) == 4 else 15.0,
                    "text": text}, None
    if kind == "box":
        values = _numbers(rest, (6,))
        if values is not None and all(lo <= hi for lo, hi in zip(values[:3], values[3:])):
            return {"kind": "box", "min": values[:3], "max": values[3:], "text": text}, None
    return None, refusal("INVALID_SELECTION", f"face_selection {text!r} is not {FACE_SELECTION_SYNTAX}")


def edge_selection(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Parse an edge selection (``EDGE_SELECTION_SYNTAX``).

    Returns:
        ``(selection, None)`` or ``(None, refusal)``.
    """
    kind, _, rest = text.strip().lower().partition(":")
    if kind == "polygroup" and not rest:
        return {"kind": "polygroup", "text": text}, None
    if kind == "sharp":
        values = _numbers(rest, (1,)) if rest else [20.0]
        if values is not None and 0 < values[0] < 180:
            return {"kind": "sharp", "angle": values[0], "text": text}, None
    return None, refusal("INVALID_SELECTION", f"edge_selection {text!r} is not {EDGE_SELECTION_SYNTAX}")


_ASSET_SEGMENT = re.compile(r"^[A-Za-z0-9_]+$")


def game_asset_path(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Check a StaticMesh destination: a package path under ``/Game``.

    Args:
        text: ``/Game/Folder/SM_Name``, optionally with its object name
            (``/Game/Folder/SM_Name.SM_Name``).

    Returns:
        ``(package path, None)``, or ``(None, refusal)``: ``SandboxError``
        (``OUTSIDE_GAME_CONTENT``) for a path outside ``/Game`` (``/Engine``,
        a plugin's mount point), ``ValidationError`` (``INVALID_ASSET_PATH``)
        for a name Unreal would rewrite or refuse.
    """
    package, _, name = text.strip().partition(".")
    if not package.startswith("/Game/"):
        return None, {
            "success": False,
            "error_type": SANDBOX_ERROR,
            "error": f"asset_path {text!r} is outside /Game: simul writes only project content, never engine or "
            "plugin content. Nothing was changed.",
            "details": {"code": "OUTSIDE_GAME_CONTENT", "asset_path": text},
        }
    segments = package[len("/Game/"):].split("/")
    if not segments or not all(_ASSET_SEGMENT.match(segment) for segment in segments) or (
        name and name != segments[-1]
    ):
        return None, refusal(
            "INVALID_ASSET_PATH",
            f"asset_path {text!r} must be /Game/<folders>/<name> with letters, digits and underscores only",
        )
    return package, None


# The attachment guard (``_unreal_attach_scripts`` verify) runs before the tool
# script and raises RuntimeError when the editor no longer matches the
# attachment; the script never starts.
_ATTACHMENT_CHANGED = re.compile(
    r"^RuntimeError: ((?:Unreal attachment changed|Attached viewport is unavailable)[^\n]*?)\s*$",
    re.MULTILINE,
)


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
    "BOOLEAN_ALIASES",
    "BOOLEAN_OPERATIONS",
    "COMPUTE_UV",
    "CONVERT",
    "CONVEX_DECOMPOSITION",
    "DEFAULT_ASSET_FOLDER",
    "CONVEX_HULL",
    "EDGE_SELECTION_SYNTAX",
    "EDIT_TOPOLOGY",
    "ENGINE_ERROR",
    "FACE_SELECTION_SYNTAX",
    "GENERATE_PRIMITIVE",
    "GEOMETRY_SCRIPT_FAILED",
    "HULL_OUTPUTS",
    "INVALID_STATE",
    "MAX_RESULT_TRIANGLES",
    "MESH_BOOLEAN",
    "MESH_HELPERS",
    "MESH_REFUSALS",
    "MESH_SCRIPT_TIMEOUT_S",
    "PLANE_CUT",
    "PRIMITIVE_DIMENSIONS",
    "PRIMITIVE_TYPES",
    "REMESH",
    "SIMPLIFY",
    "STATIC_MESH_COLLISION",
    "SUBDIVIDE",
    "SUBDIVISION_SCHEMES",
    "TARGET_FORMATS",
    "TOPOLOGY_OPERATIONS",
    "UNSUPPORTED_OPERATION",
    "UV_METHODS",
    "VALIDATE",
    "VALIDATION_CHECKS",
    "XATLAS_MAX_TRIANGLES",
    "attachment_refusal",
    "edge_selection",
    "face_selection",
    "game_asset_path",
    "invalid_choice",
    "mesh_script",
    "primitive_dimensions",
    "refusal",
]
