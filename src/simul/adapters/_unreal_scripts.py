"""Shared, fixed Python helpers executed inside the Unreal editor.

Caller data is embedded with repr/JSON, never interpolated as Python source.
These use the Python API so UObject references and world transforms do not
depend on Remote Control's property-description wire format.
"""

from typing import Any, Dict

#: ``error_type`` of an editor-world change refused because Play In Editor (or
#: Simulate In Editor) is running.
PIE_ACTIVE = "PIEActive"

#: The refusal's ``error``; ``{action}`` names the change, e.g. "spawn actors".
PIE_ACTIVE_ERROR = (
    "Play In Editor is running, and Unreal's editor cannot {action} until it stops"
)

#: The refusal's ``details.hint``.
PIE_STOP_HINT = (
    "Stop PIE with control_unreal_simulation, action 'stop' (full tool surface; CLI: simul unreal sim "
    "stop), then retry. Reading the level, and editing actors already in it by path, keep working "
    "while PIE runs."
)


def pie_refusal(action: str) -> Dict[str, Any]:
    """The error envelope of an editor-world change Unreal refuses while PIE runs.

    Args:
        action: What was refused, completing "cannot ...": "spawn actors".

    Returns:
        ``success`` False, ``error_type`` PIEActive, the error and a hint naming
        the tool that stops PIE.
    """
    return {
        "success": False,
        "error_type": PIE_ACTIVE,
        "error": PIE_ACTIVE_ERROR.format(action=action),
        "details": {"hint": PIE_STOP_HINT},
    }


# The editor world while Play In Editor (or Simulate In Editor) runs. Unreal's
# editor scripting then refuses (observed on UE 5.7.4):
# UnrealEditorSubsystem.get_editor_world() returns None, and
# EditorActorSubsystem spawns and destroys nothing, lists no actors and selects
# none. LayersSubsystem.get_world() returns GWorld, which is the editor world
# whenever Remote Control runs a script: the editor points GWorld at a play
# world only while it ticks that world, and checks at the start of every editor
# tick that it is back on the editor world. That holds after the game opens
# another level too, when the play world's package (UEDPIE_<n>_<map>) no longer
# names the editor's map. GWorld holding a play world means no editor world can
# be determined, and editor_world() returns None.
EDITOR_WORLD_FUNCTION = """
def editor_world():
    editor = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    world = editor.get_editor_world()
    if world is not None or editor.get_game_world() is None:
        return world
    world = unreal.get_editor_subsystem(unreal.LayersSubsystem).get_world()
    if world is None or re.match("UEDPIE_[0-9]+_", world.get_outermost().get_name().rpartition("/")[2]):
        return None
    return world
"""

# PIE-aware helpers for the scripts below: pie_active(), pie_refusal(action)
# (the same envelope as the Python function), editor_world() and
# level_actors(). During PIE, level_actors() reads the editor world directly:
# every editable actor of its visible levels except the WorldSettings, which
# can include the builder brush and helper actors the Outliner hides.
EDITOR_WORLD = (
    "import json, re, unreal\n"
    f"_PIE_ACTIVE, _PIE_ERROR, _PIE_HINT = {PIE_ACTIVE!r}, {PIE_ACTIVE_ERROR!r}, {PIE_STOP_HINT!r}\n"
    + EDITOR_WORLD_FUNCTION
    + """
def pie_active():
    return bool(unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor())

def pie_refusal(action):
    return {"success": False, "error_type": _PIE_ACTIVE, "error": _PIE_ERROR.format(action=action),
            "details": {"hint": _PIE_HINT}}

def level_actors():
    if not pie_active():
        return list(unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors())
    world = editor_world()
    if world is None:
        return []
    return [actor for actor in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
            if actor.is_editable() and not isinstance(actor, unreal.WorldSettings)]
"""
)

#: Message of the ``SimulNotFound`` an editor script raises for an actor path
#: that names no live actor of the editor world; ``{path}`` is the path.
ACTOR_NOT_FOUND = "Actor not found: {path}"

# actor_at(path) is the live actor at ``path`` in the editor world, PIE-aware
# through editor_world(). It finds the actor among loaded objects and never
# loads one: load_object on the path of an actor in a map that is not open
# loads that map's package from disk, so an edit landed in a map nobody sees
# and left it dirty for the next Save All (UE 5.7.4). A destroyed actor stays
# findable, invalid (SystemLibrary.is_valid is False), until garbage
# collection; an actor of a map loaded but not open, or of the PIE world,
# belongs to another world. Each raises SimulNotFound, whose message starts
# with ACTOR_NOT_FOUND, and the session reports it as NotFoundError.
# actor_at(path, play_world=True) also takes an actor of the running Play In
# Editor world (UnrealEditorSubsystem.get_game_world), for the physics and
# collision edits that change the running copy on purpose.
ACTOR_HELPERS = EDITOR_WORLD + f"_ACTOR_NOT_FOUND = {ACTOR_NOT_FOUND!r}\n" + """
class SimulNotFound(Exception):
    pass

def actor_at(path, play_world=False):
    actor = unreal.find_object(None, path)
    world = editor_world()
    if not isinstance(actor, unreal.Actor) or not unreal.SystemLibrary.is_valid(actor):
        where = world.get_path_name() if world is not None else "no editor world"
        raise SimulNotFound(_ACTOR_NOT_FOUND.format(path=path)
                            + " (no live actor at that path in the level open in the editor, " + where + ")")
    owner = actor.get_world()
    if owner is not None and owner == world:
        return actor
    if (play_world and owner is not None
            and owner == unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()):
        return actor
    raise SimulNotFound(_ACTOR_NOT_FOUND.format(path=path) + " (it belongs to "
                        + (owner.get_path_name() if owner is not None else "no world")
                        + ", not to the level open in the editor, "
                        + (world.get_path_name() if world is not None else "none")
                        + "; open that map to edit it)")

def actor_transform(actor):
    location = actor.get_actor_location()
    rotation = actor.get_actor_rotation()
    scale = actor.get_actor_scale3d()
    return {
        "location": [location.x, location.y, location.z],
        "rotation": [rotation.pitch, rotation.yaw, rotation.roll],
        "scale": [scale.x, scale.y, scale.z],
    }

def actor_info(actor):
    root = actor.root_component
    return {
        "name": actor.get_actor_label(),
        "path": actor.get_path_name(),
        "class_name": actor.get_class().get_name(),
        **actor_transform(actor),
        "tags": [str(tag) for tag in actor.get_editor_property("tags")],
        "components": [
            {"name": c.get_name(), "class_name": c.get_class().get_name(),
             "is_root": c == root}
            for c in actor.get_components_by_class(unreal.ActorComponent)
        ],
        "mobility": root.mobility.name.title() if root else "None",
        "is_hidden": bool(actor.get_editor_property("hidden")),
    }

def actor_component(actor, component_class):
    root = actor.root_component
    if isinstance(root, component_class):
        return root
    components = actor.get_components_by_class(component_class)
    if len(components) != 1:
        raise ValueError("Actor must have a compatible root or exactly one "
                         + component_class.__name__ + ": " + actor.get_path_name())
    return components[0]
"""

SIMULATION_STATUS = """
import json, unreal
editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
playing = editor.is_in_play_in_editor()
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world() if playing else None
if playing and world is None:
    raise RuntimeError("PIE is active but its game world is not available")
print(json.dumps({
    "is_playing": playing,
    "is_paused": unreal.GameplayStatics.is_game_paused(world) if world else False,
    "frame_count": None,
    "sim_time": unreal.GameplayStatics.get_time_seconds(world) if world else 0.0,
}))
"""

# Both USD scripts keep every editor object they create inside a function, so
# nothing outlives the script in the console namespace Remote Control runs it
# in (FileExecutionScope Public): a module-level AssetExportTask kept its
# Object, the editor world, referenced through Python's reference collector,
# and the next level load failed the editor's world leak check, "Old World ...
# not cleaned up by GC", and took the editor down (observed on UE 5.7.4). The
# import's factory holds the scene actor it spawned the same way. Each task's
# object references are also cleared before the function returns.

# Expects ``args`` = {"path", "destination", "name", "options", "replace"}:
# the USD file on the editor host, the content folder, the folder name the
# import is filed under (the file's name, sanitized as Unreal names assets),
# the importer options by UsdStageImportOptions property name, and whether to
# replace what an earlier import left. Imported actors are found as the level
# actors that were not there before, a listing Unreal withholds while PIE
# runs, so the import is refused then.
#
# The options are set on a fresh UsdStageImportOptions without change
# notifications. UsdStageImportOptions saves its config on every notified
# change (PostEditChangeProperty calls SaveConfig, UE 5.7), and
# set_editor_property notifies by default, so one import with import_actors
# false wrote bImportActors=False to the project's
# EditorPerProjectUserSettings.ini: every later import, the editor's own
# "Import into Level" dialog included, then spawned no actors. A fresh object
# starts from those saved settings, so the options simul documents are always
# set, to Unreal's defaults unless given. A saved setting that differs from
# Unreal's default is reported in warnings and left as it is: nothing tells a
# setting simul once wrote from one the user chose.
#
# Without replace the import is refused when the folder already holds assets,
# and runs with Unreal's Append actor policy (new actors beside existing ones)
# and Ignore asset policy (an existing asset is kept), so nothing is replaced.
# Unreal's defaults, Replace for both, replaced an earlier import's assets
# (a customized mesh lost its collision and lightmap resolution) and deleted
# its actors even though AssetImportTask.replace_existing was False, which the
# USD importer does not read. With replace, both policies are Replace.
USD_IMPORT = EDITOR_WORLD + """
import os

# UsdStageImportOptions' constructor defaults (UE 5.7) for the options simul sets on every import.
DEFAULTS = {"import_actors": True, "import_geometry": True, "import_materials": True}
LISTED = 20

def import_options(values, replace):
    options = unreal.UsdStageImportOptions()
    quiet = unreal.PropertyAccessChangeNotifyMode.NEVER
    for key, value in dict(DEFAULTS, **values).items():
        options.set_editor_property(key, value, quiet)
    if replace:
        actors, assets = unreal.ReplaceActorPolicy.REPLACE, unreal.ReplaceAssetPolicy.REPLACE
    else:
        actors, assets = unreal.ReplaceActorPolicy.APPEND, unreal.ReplaceAssetPolicy.IGNORE
    options.set_editor_property("existing_actor_policy", actors, quiet)
    options.set_editor_property("existing_asset_policy", assets, quiet)
    return options

def saved_settings():
    # The class default object holds the project's saved import settings.
    saved = unreal.get_default_object(unreal.UsdStageImportOptions)
    return {key: saved.get_editor_property(key) for key in DEFAULTS}

def import_usd(path, destination, name, options, replace):
    if not hasattr(unreal, "UsdStageImportFactory"):
        return {"success": False, "error_type": "PluginUnavailable",
                "error": "USD import requires the USDImporter plugin enabled at editor startup."}
    if pie_active():
        return pie_refusal("import USD into the level")
    if not os.path.isfile(path):
        return {"success": False, "error_type": "NotFoundError",
                "error": "USD file not found on the editor host: " + path}
    folder = destination + "/" + name
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    existing = sorted({str(asset.package_name) for asset in registry.get_assets_by_path(folder, recursive=True)})
    if existing and not replace:
        return {"success": False, "error_type": "InvalidState",
                "error": (folder + " already holds " + str(len(existing)) + " asset(s) and replace_existing is "
                          "false, so nothing was imported"),
                "details": {"folder": folder, "existing_assets": existing[:LISTED],
                            "hint": "Pass another destination_path, or replace_existing=true to replace "
                                    "those assets and the level's actors from an earlier import of this file"}}
    saved_before = saved_settings()
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    before = [actor.get_path_name() for actor in subsystem.get_all_level_actors()]
    task = unreal.AssetImportTask()
    try:
        task.filename = path
        task.destination_path = destination
        task.destination_name = name
        task.automated = True
        task.save = True
        task.replace_existing = replace
        task.factory = unreal.UsdStageImportFactory()
        task.options = import_options(options, replace)
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        assets = [str(p) for p in task.imported_object_paths]
    finally:
        task.factory = None
        task.options = None
    after = [actor.get_path_name() for actor in subsystem.get_all_level_actors()]
    known = set(before)
    actors = [p for p in after if p not in known]
    current = set(after)
    removed = [p for p in before if p not in current]
    if not assets and not actors:
        return {"success": False, "error_type": "EngineError",
                "error": "The USD importer produced no assets or actors from " + path + "; see the editor log"}
    warnings = []
    saved_after = saved_settings()
    if saved_after != saved_before:
        warnings.append("The import changed this project's saved USD import settings: "
                        + json.dumps(saved_before) + " became " + json.dumps(saved_after))
    differing = {key: value for key, value in saved_after.items() if value != DEFAULTS[key]}
    if differing:
        warnings.append(
            "This project's saved USD import settings differ from Unreal's defaults ("
            + ", ".join(key + "=" + json.dumps(value) for key, value in sorted(differing.items()))
            + "); they are in Saved/Config/<Platform>Editor/EditorPerProjectUserSettings.ini, section "
            "[/Script/USDStageImporter.UsdStageImportOptions], and the editor's own USD import dialog uses them. "
            "simul set these options explicitly for this import and left the saved settings unchanged. "
            "Earlier simul versions saved the options an import was given, so an import with "
            "import_actors false left bImportActors=False there.")
    return {"imported_assets": assets, "actor_paths": actors, "removed_actor_paths": removed,
            "warnings": warnings}

print(json.dumps(import_usd(**args)))
"""

# Expects ``args`` = {"path", "actors", "options"}. The export selects the
# actors and exports the editor world, neither of which Unreal allows while PIE
# runs, so the export is refused then. Each actor path must name an actor of
# the level being edited (actor_at). Unreal's USD level exporter writes
# what it accepts and leaves out the rest without an error (actors of another
# world, editor helpers such as a WorldSettings or PlayerStart), so the actors
# exported are read back from the written stage: the prim each actor's root
# component went to (UsdConversionLibrary.get_prim_path_for_object, which the
# exporter names it by, with the exporter's _<n> suffix for a name already
# used). An export that wrote none of them fails.
USD_EXPORT = ACTOR_HELPERS + """
import os

def exported_prims(path, actors, inner):
    from pxr import Usd
    stage = Usd.Stage.Open(path, Usd.Stage.LoadNone)
    claimed, exported, skipped = set(), [], []
    try:
        for actor in actors:
            root = actor.root_component
            found = None
            if root is not None:
                base = str(unreal.UsdConversionLibrary.get_prim_path_for_object(
                    root, "", inner.export_actor_folders, inner.root_prim_name))
                for candidate in [base] + [base + "_" + str(n) for n in range(len(actors))]:
                    if candidate not in claimed and stage.GetPrimAtPath(candidate):
                        found = candidate
                        break
            if found is None:
                skipped.append(actor.get_path_name())
            else:
                claimed.add(found)
                exported.append({"actor_path": actor.get_path_name(), "prim_path": found})
    finally:
        del stage
    return exported, skipped

def export_usd(path, actor_paths, values):
    if not hasattr(unreal, "LevelExporterUSD"):
        return {"success": False, "error_type": "PluginUnavailable",
                "error": "USD export requires the USDImporter plugin enabled at editor startup."}
    if pie_active():
        return pie_refusal("export actors to USD")
    actors = [actor_at(actor_path) for actor_path in actor_paths]
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    previous_selection = subsystem.get_selected_level_actors()
    # UE's native level exporter hands its task to Python through the CDO.
    options = unreal.get_default_object(unreal.LevelExporterUSDOptions)
    previous_inner = options.inner.copy()
    inner = previous_inner.copy()
    for key, value in values.items():
        inner.set_editor_property(key, value)
    inner.selection_only = True
    # Secondary assets must stay under the authorized output directory too.
    inner.asset_folder = unreal.DirectoryPath(os.path.dirname(path))
    task = unreal.AssetExportTask()
    try:
        task.object = editor_world()
        task.filename = path
        task.automated = True
        task.prompt = False
        task.selected = True
        task.replace_identical = True
        task.exporter = unreal.LevelExporterUSD()
        task.options = options
        options.inner = inner
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subsystem.set_selected_level_actors(actors)
        if not unreal.Exporter.run_asset_export_task(task):
            return {"success": False, "error_type": "EngineError",
                    "error": "USD export failed: " + ("; ".join(str(e) for e in task.errors) or "see the editor log")}
    finally:
        task.object = None
        task.exporter = None
        task.options = None
        subsystem.set_selected_level_actors(previous_selection)
        options.inner = previous_inner
    size = os.path.getsize(path) if os.path.isfile(path) else 0
    if size == 0:
        return {"success": False, "error_type": "EngineError", "error": "USD export wrote no file at " + path}
    exported, skipped = exported_prims(path, actors, inner)
    result = {"output_path": path, "actors_exported": len(exported), "exported_prims": exported,
              "skipped_actors": skipped, "file_size_bytes": size}
    if not exported:
        result.update(success=False, error_type="EngineError",
                      error="Unreal's USD exporter wrote no prim for any actor given (it leaves out editor "
                            "helpers such as WorldSettings, PlayerStart and brushes): " + ", ".join(skipped))
    return result

print(json.dumps(export_usd(args["path"], args["actors"], args["options"])))
"""

INTERCHANGE_INFO = """
import json, unreal
if not hasattr(unreal, "InterchangeManager"):
    print(json.dumps({"success": False, "error_type": "PluginUnavailable",
                      "error": "Interchange is not available in this editor"}))
else:
    manager = unreal.InterchangeManager.get_interchange_manager_scripted()
    formats = manager.get_supported_formats(unreal.InterchangeTranslatorType.SCENES)
    print(json.dumps({
        "pipelines": [], "pipeline_enumeration_available": False,
        "supported_formats": sorted(set(str(f) for f in formats)),
        "interchange_version": unreal.SystemLibrary.get_engine_version(),
        "usd_import_available": hasattr(unreal, "UsdStageImportFactory"),
        "usd_export_available": hasattr(unreal, "LevelExporterUSD"),
        "simready_available": False,
    }))
"""

# Engine/project metadata in one Remote Control round trip, shared by
# health_check and get_engine_info.
ENGINE_METADATA = EDITOR_WORLD + """
world = editor_world()
print(json.dumps({
    "engine_version": unreal.SystemLibrary.get_engine_version(),
    "project_name": unreal.SystemLibrary.get_game_name(),
    "loaded_map": world.get_path_name() if world else "",
    "platform": unreal.GameplayStatics.get_platform_name(),
    # project_dir() is relative to the engine binaries; report the real path.
    "project_dir": unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()),
    "pie_active": pie_active(),
}))
"""

# The editor world's persistent level, also while PIE runs.
LOADED_MAP = EDITOR_WORLD + """
world = editor_world()
if world is None:
    raise RuntimeError("No editor world is loaded")
print(json.dumps({"map_path": world.get_path_name(), "pie_active": pie_active()}))
"""

# Expects ``args`` = {"actor_path", "values"}, where ``values`` holds only the
# components to write: "location" [x, y, z] cm, "rotation" [pitch, yaw, roll]
# degrees, "scale" [x, y, z]. A component left out is not touched, so it keeps
# its current value. Writes in one undoable transaction (Actor.modify also
# records the root component), then reads the whole world transform back.
SET_ACTOR_TRANSFORM = ACTOR_HELPERS + """
actor = actor_at(args["actor_path"])
values = args["values"]
if values:
    with unreal.ScopedEditorTransaction("simul: set actor transform"):
        actor.modify()
        if "location" in values:
            actor.set_actor_location(unreal.Vector(*values["location"]), False, True)
        if "rotation" in values:
            pitch, yaw, roll = values["rotation"]
            actor.set_actor_rotation(unreal.Rotator(pitch=pitch, yaw=yaw, roll=roll), True)
        if "scale" in values:
            actor.set_actor_scale3d(unreal.Vector(*values["scale"]))
print(json.dumps({"actor_path": actor.get_path_name(), **actor_transform(actor)}))
"""

# Expects ``args`` = {"actor_path"}. Adds the actor to the editor selection,
# reads the selection back and prints the actor's location and bounds for the
# camera. The bounds are AActor::GetActorBounds over every primitive component,
# its child actors' included, as a center and a half-size. The half-size is
# zero when no primitive component has a size (a StaticMeshActor without a
# mesh); the center is then where such a component sits, or the world origin
# when the actor has none. The editor-only components the editor's own Focus
# leaves out are left out too, since their bounds can reach far beyond the
# actor (UE 5.7 source, UE 5.7.4). Its Focus ignores the bounds of some
# (GetIgnoreBoundsForEditorFocus): the frustum the editor draws for a camera
# or a scene capture, a cube reaching the frustum's far end, 10 m, on every
# side of it, and a CineCamera's debug focus plane, 100 m across at the focus
# distance (1 km by default), which turning the plane off only hides. It never
# visits those not attached under their actor's root, such as the spare icon a
# SkyLight or a PlanarReflection leaves at the world origin. Its Focus also
# ignores every shape (a box, sphere or capsule component); a shape whose
# collision is turned off, which collides and overlaps with nothing, is left
# out as well, editor-only or not. The influence sphere of a
# SphereReflectionCapture (30 m in radius by default) and the boxes of a
# BoxReflectionCapture are such shapes, which Unreal does not make editor-only;
# a trigger's shape collides (the Trigger profile). The box is then
# drawn around the bounds of the other primitive components, provided
# GetActorBounds is the box around all of them: GetActorBounds is virtual, and
# an actor reporting other bounds, such as a level instance (its level's
# content) or a NavLinkProxy (its links), keeps them, as does an actor with
# nothing but such components. Any other component that is not editor-only
# counts, such as a trigger box's collision shape, which the editor's Focus
# ignores.
# While Play In Editor (or Simulate) runs nothing is selected: Unreal's editor
# reports no selection then and clears it when PIE stops (observed on UE
# 5.7.4), so a selection made during PIE would be lost unseen. Outside PIE an
# actor Unreal will not select (hidden in the editor or in a hidden or locked
# level, a child actor or an actor inside a level instance, which Unreal
# selects only through its parent, or any actor while the editor mode or a
# selection lock prevents selection) reads back unselected.
FOCUS_ACTOR = ACTOR_HELPERS + """
import math

def left_out_by_editor_focus(component, root):
    if (isinstance(component, unreal.ShapeComponent)
            and component.get_collision_enabled() == unreal.CollisionEnabled.NO_COLLISION):
        return True
    if not component.get_editor_property("is_editor_only"):
        return False
    if component.get_ignore_bounds_for_editor_focus():
        return True
    while component is not None and component != root:
        component = component.get_attach_parent()
    return component is None

def box_around(components):
    low, high = [float("inf")] * 3, [float("-inf")] * 3
    for component in components:
        center, half, _ = unreal.SystemLibrary.get_component_bounds(component)
        for axis, (mid, size) in enumerate(((center.x, half.x), (center.y, half.y), (center.z, half.z))):
            low[axis], high[axis] = min(low[axis], mid - size), max(high[axis], mid + size)
    return ([(lo + hi) / 2 for lo, hi in zip(low, high)],
            [(hi - lo) / 2 for lo, hi in zip(low, high)])

def focus_bounds(actor):
    origin, extent = actor.get_actor_bounds(only_colliding_components=False, include_from_child_actors=True)
    reported = ([origin.x, origin.y, origin.z], [extent.x, extent.y, extent.z])
    actors = [actor] + list(actor.get_all_child_actors())
    components = [(c, a.root_component) for a in actors for c in a.get_components_by_class(unreal.PrimitiveComponent)]
    kept = [c for c, root in components if not left_out_by_editor_focus(c, root)]
    if not kept or len(kept) == len(components):
        return reported
    every = box_around([c for c, _ in components])
    if not all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-3)
               for a, b in zip(every[0] + every[1], reported[0] + reported[1])):
        return reported
    return box_around(kept)

actor = actor_at(args["actor_path"])
pie = pie_active()
selected = False
if not pie:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    subsystem.set_actor_selection_state(actor, True)
    selected = actor in subsystem.get_selected_level_actors()
origin, extent = focus_bounds(actor)
print(json.dumps({"location": actor_transform(actor)["location"],
                  "bounds_origin": origin, "bounds_extent": extent,
                  "selected": selected, "pie_active": pie}))
"""

# Expects ``args`` = {"path", "location", "rotation", "label"}: location
# [x, y, z] cm, rotation [pitch, yaw, roll] degrees, label None to keep the
# default. An Actor class (``/Script/Engine.PointLight``, a Blueprint's
# ``..._C`` class) spawns through spawn_actor_from_class. Any other asset goes
# through spawn_actor_from_object, the actor factory the editor uses when the
# asset is dropped into the level: a StaticMesh becomes a StaticMeshActor
# showing it, a Blueprint asset its actor. Spawn and label are one undoable
# transaction. When the path loads nothing, or Unreal places no actor for it,
# the script prints an error instead of an empty actor path; while PIE runs,
# when Unreal places nothing whatever the path, it prints the PIE refusal.
SPAWN_ACTOR = ACTOR_HELPERS + """
def spawn(path, location, rotation, label):
    if pie_active():
        return pie_refusal("spawn actors")
    source = unreal.load_object(None, path)
    if source is None:
        return {"success": False, "error_type": "NotFoundError",
                "error": "No class or asset at " + repr(path) + ": pass an Actor class path "
                         "such as /Script/Engine.PointLight or an asset path such as "
                         "/Engine/BasicShapes/Cube.Cube"}
    if isinstance(source, unreal.Actor):
        return {"success": False, "error_type": "SpawnError",
                "error": path + " is an actor in a level, not a class or asset"}
    is_class = isinstance(source, unreal.Class)
    if is_class and not unreal.MathLibrary.class_is_child_of(source, unreal.Actor.static_class()):
        return {"success": False, "error_type": "SpawnError",
                "error": path + " is a class but not an Actor class"}
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    location = unreal.Vector(*location)
    pitch, yaw, roll = rotation
    rotation = unreal.Rotator(pitch=pitch, yaw=yaw, roll=roll)
    with unreal.ScopedEditorTransaction("simul: spawn actor"):
        if is_class:
            actor = subsystem.spawn_actor_from_class(source, location, rotation)
        else:
            actor = subsystem.spawn_actor_from_object(source, location, rotation)
        if actor is not None and label is not None:
            actor.set_actor_label(label)
    if actor is None:
        return {"success": False, "error_type": "SpawnError",
                "error": "Unreal placed no actor for " + path + " (a "
                         + source.get_class().get_name() + "); pass an Actor class or an "
                         "asset the editor can place, such as a StaticMesh or a Blueprint"}
    transform = actor_transform(actor)
    return {"actor_path": actor.get_path_name(), "actor_class": actor.get_class().get_name(),
            "label": actor.get_actor_label(), "location": transform["location"],
            "rotation": transform["rotation"]}

print(json.dumps(spawn(**args)))
"""

# Expects ``args`` = {"actor_path", "parent_path"}; parent_path None detaches.
# K2_AttachToActor (Python attach_to_actor) returns False and changes nothing
# for an attachment Unreal refuses: the actor itself, a parent attached under
# the child, or a Static child under a parent that is not Static (UE 5.7.4).
# The first two and the mobility rule are refused here first with their
# reason; a parent path that names no live actor of the editor world raises
# SimulNotFound like the child's. The attachment keeps the world transform
# (location, rotation and scale) and is one undoable transaction, with the
# child, the new parent and the previous parent recorded first (Actor.modify
# also records the root component, whose attachment changes). The parent is
# read back; an attachment Unreal refused cancels the transaction.
SET_ACTOR_PARENT = ACTOR_HELPERS + """
def path_of(actor):
    return actor.get_path_name() if actor is not None else None

def refused(error_type, message):
    return {"success": False, "error_type": error_type, "error": message}

def result(actor_path, parent, previous):
    return {"actor_path": actor_path, "parent_path": path_of(parent),
            "previous_parent_path": path_of(previous), "changed": parent != previous}

def detach(actor_path, child, previous):
    if previous is not None:
        rule = unreal.DetachmentRule.KEEP_WORLD
        with unreal.ScopedEditorTransaction("simul: detach actor"):
            child.modify()
            previous.modify()
            child.detach_from_actor(rule, rule, rule)
    after = child.get_attach_parent_actor()
    if after is not None:
        return refused("EngineError", "Unreal did not detach " + actor_path + " from " + path_of(after))
    return result(actor_path, None, previous)

def set_parent(actor_path, parent_path):
    child = actor_at(actor_path)
    previous = child.get_attach_parent_actor()
    if parent_path is None:
        return detach(actor_path, child, previous)
    parent = actor_at(parent_path)
    if parent == child:
        return refused("ValidationError", "An actor cannot be attached to itself: " + actor_path)
    node = parent
    while node is not None:
        if node == child:
            return refused("ValidationError", "Attaching " + actor_path + " to " + parent_path
                           + " would make a cycle: " + parent_path + " is attached under " + actor_path
                           + "; detach " + parent_path + " first")
        node = node.get_attach_parent_actor()
    child_root, parent_root = child.root_component, parent.root_component
    if child_root is None or parent_root is None:
        return refused("InvalidState", (actor_path if child_root is None else parent_path)
                       + " has no root component, which an attachment joins")
    static = unreal.ComponentMobility.STATIC
    if child_root.mobility == static and parent_root.mobility != static:
        return refused("InvalidState", "Unreal attaches a Static actor only under a Static parent: " + actor_path
                       + " is Static and " + parent_path + " is " + parent_root.mobility.name.title()
                       + "; make the child's root component Movable first")
    if previous == parent:
        return result(actor_path, parent, previous)
    rule = unreal.AttachmentRule.KEEP_WORLD
    with unreal.ScopedEditorTransaction("simul: attach actor") as transaction:
        child.modify()
        parent.modify()
        if previous is not None:
            previous.modify()
        attached = child.attach_to_actor(parent, "", rule, rule, rule, False)
        after = child.get_attach_parent_actor()
        if not attached and after == previous:
            transaction.cancel()
    if not attached or after != parent:
        return refused("EngineError", "Unreal did not attach " + actor_path + " to " + parent_path
                       + "; the editor log says why")
    return result(actor_path, after, previous)

print(json.dumps(set_parent(**args)))
"""

# Expects ``args`` = {"actor_path", "component_class", "component_name",
# "parent_component"}. Adds an instance component the way the editor's Add
# Component button does, through SubobjectDataSubsystem.add_new_subobject:
# Remote Control refuses AddComponentByClass on UE 5.7 ("deprecated or
# unavailable remotely"). component_class is a class path
# (/Script/Engine.PointLightComponent, a Blueprint's /Game/BP_Lamp.BP_Lamp_C or
# its /Game/BP_Lamp.BP_Lamp) or a native class name (PointLightComponent).
# A path inside a map's package is refused before anything loads. An abstract
# class is refused too: the subsystem adds one without complaint and the
# engine fires an ensure (UE 5.7.4). unreal.new_object refuses an abstract
# class before creating anything and makes a throwaway transient object of a
# concrete one, which the garbage collector reclaims; nothing else reports a
# class's abstract flag to Python. A scene component attaches under
# parent_component (a scene component of the actor, by name) or the root
# component, and becomes the root of an actor without one; any other component
# takes no parent. The add and the name are one undoable transaction; a name
# already used inside the actor is refused before it. The component is read
# back from the actor.
ADD_COMPONENT = ACTOR_HELPERS + """
def refused(error_type, message):
    return {"success": False, "error_type": error_type, "error": message}

def handle_object(handle):
    if not unreal.SubobjectDataBlueprintFunctionLibrary.is_handle_valid(handle):
        return None
    data = unreal.SubobjectDataBlueprintFunctionLibrary.get_data(handle)
    return unreal.SubobjectDataBlueprintFunctionLibrary.get_associated_object(data)

def resolve(name):
    if "/" not in name:
        wrapper = getattr(unreal, name, None)
        if isinstance(wrapper, type) and hasattr(wrapper, "static_class"):
            return wrapper.static_class(), None
        return None, ("No class named " + repr(name)
                      + ": pass a native class name such as PointLightComponent or a class path")
    if ":" in name:
        return None, name + " names an object inside a package, not a class"
    if name.startswith("/Script/"):
        found = unreal.find_object(None, name)
    else:
        package = name.split(".", 1)[0]
        assets = unreal.AssetRegistryHelpers.get_asset_registry().get_assets_by_package_name(package)
        if any(str(asset.asset_class_path.asset_name) == "World" for asset in assets):
            return None, name + " is in a map package, not a class; simul loads no map"
        found = unreal.load_object(None, name)
    if isinstance(found, unreal.Blueprint):
        found = found.generated_class()
    if not isinstance(found, unreal.Class):
        return None, "No class at " + repr(name)
    return found, None

def describe(actor, component):
    parent = component.get_attach_parent() if isinstance(component, unreal.SceneComponent) else None
    return {"component_path": component.get_path_name(), "component_name": component.get_name(),
            "component_class": component.get_class().get_name(),
            "parent_component": parent.get_name() if parent is not None else None,
            "is_root": component == actor.root_component}

def add(actor_path, component_class, component_name, parent_component):
    actor = actor_at(actor_path)
    cls, reason = resolve(component_class)
    if cls is None:
        return refused("NotFoundError", reason)
    if not unreal.MathLibrary.class_is_child_of(cls, unreal.ActorComponent.static_class()):
        return refused("ValidationError", component_class + " is the class " + cls.get_name()
                       + ", not a component class (an ActorComponent)")
    try:
        unreal.new_object(cls)
    except Exception as exc:
        if "abstract" not in str(exc):
            raise
        return refused("ValidationError", cls.get_name() + " is abstract; pass a concrete component class")
    scene = unreal.MathLibrary.class_is_child_of(cls, unreal.SceneComponent.static_class())
    if component_name is not None and unreal.find_object(actor, component_name) is not None:
        return refused("ValidationError", actor_path + " already has a component or subobject named "
                       + repr(component_name))
    target = actor.root_component if scene else None
    if parent_component is not None:
        if not scene:
            return refused("ValidationError", cls.get_name() + " is not a scene component, so it attaches to "
                           "nothing and takes no parent_component")
        target = next((c for c in actor.get_components_by_class(unreal.SceneComponent)
                       if c.get_name() == parent_component), None)
        if target is None:
            return refused("NotFoundError", "Component not found: " + parent_component
                           + " (no scene component of that name on " + actor_path + ")")
    subsystem = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    handles = list(subsystem.k2_gather_subobject_data_for_instance(actor))
    parent_handle = next((h for h in handles if target is not None and handle_object(h) == target), handles[0])
    with unreal.ScopedEditorTransaction("simul: add component") as transaction:
        actor.modify()
        params = unreal.AddNewSubobjectParams(parent_handle=parent_handle, new_class=cls, blueprint_context=None)
        handle, failure = subsystem.add_new_subobject(params)
        component = handle_object(handle)
        if component is None:
            transaction.cancel()
            return refused("EngineError", "Unreal added no " + cls.get_name() + " to " + actor_path
                           + (": " + str(failure) if str(failure) else ""))
        if component_name is not None and not (subsystem.rename_subobject(handle, unreal.Text(component_name))
                                               and component.get_name() == component_name):
            subsystem.delete_subobject(handles[0], handle, None)
            transaction.cancel()
            return refused("EngineError", "Unreal did not name the new " + cls.get_name() + " "
                           + repr(component_name) + ", so it was removed again")
    if component not in actor.get_components_by_class(unreal.ActorComponent):
        return refused("EngineError", "The new " + cls.get_name() + " is not among " + actor_path + "'s components")
    return dict(describe(actor, component), actor_path=actor_path)

print(json.dumps(add(**args)))
"""

# Expects ``args`` = {"label", "match_mode", "max_results"}. Semantic labels
# are actor tags. Matching ignores case, as Unreal's tag comparison (FName)
# does: "exact" compares whole tags, "contains" looks for the label inside a
# tag and "regex" searches each tag for the regular expression. Every match is
# counted; the first max_results are returned as actor_info entries. While PIE
# runs the editor world is searched (level_actors) and pie_active says so.
FIND_ACTORS_BY_TAG = ACTOR_HELPERS + EDITOR_WORLD + """
label, mode = args["label"], args["match_mode"]
if mode == "regex":
    pattern = re.compile(label, re.IGNORECASE)

def matches(tag):
    if mode == "regex":
        return pattern.search(tag) is not None
    if mode == "contains":
        return label.lower() in tag.lower()
    return tag.lower() == label.lower()

actors = []
total = 0
for actor in level_actors():
    if any(matches(str(tag)) for tag in actor.get_editor_property("tags")):
        total += 1
        if len(actors) < args["max_results"]:
            actors.append(actor_info(actor))
print(json.dumps({"actors": actors, "total_matches": total,
                  "truncated": total > len(actors), "label_searched": label,
                  "pie_active": pie_active()}))
"""

# Scene digest in one round trip. Class keys are UClass path names
# (``/Script/Engine.StaticMeshActor``), the same strings Remote Control's
# ``/remote/object/describe`` reports as ``Class``. Static meshes, lights and
# cameras are counted by class, subclasses included: a light is a Light actor
# (directional, point, spot, rect) or a SkyLight, which is not a Light; a
# camera is a CameraActor (CineCameraActor included). A class name containing
# "Light" or "Camera" (LightmassImportanceVolume, CameraBlockingVolume) is
# neither.
SCENE_SUMMARY = EDITOR_WORLD + """
world = editor_world()
counts = {}
total = static_meshes = lights = cameras = 0
for actor in level_actors():
    name = actor.get_class().get_path_name()
    counts[name] = counts.get(name, 0) + 1
    total += 1
    static_meshes += isinstance(actor, unreal.StaticMeshActor)
    lights += isinstance(actor, (unreal.Light, unreal.SkyLight))
    cameras += isinstance(actor, unreal.CameraActor)
print(json.dumps({
    "map_path": world.get_path_name() if world else "",
    "total_actors": total,
    "actor_class_counts": counts,
    "static_meshes": static_meshes,
    "lights": lights,
    "cameras": cameras,
    "pie_active": pie_active(),
}))
"""

# Expects ``args`` = {"parent", "name", "folder"}. EditorAssetLibrary loads the
# parent, checks the destination and saves the instance, and while PIE runs it
# loads, finds and saves no assets (observed on UE 5.7.4: load_asset returns
# None and does_asset_exist False for assets that exist, and the editor logs
# "The Editor is currently in a play mode"). So the script prints the PIE
# refusal then, before anything is created, rather than a parent not found.
CREATE_MATERIAL_INSTANCE = EDITOR_WORLD + """
def create(parent, name, folder):
    if pie_active():
        return pie_refusal("create material instances")
    material = unreal.EditorAssetLibrary.load_asset(parent)
    destination = folder + "/" + name
    if material is None or not isinstance(material, unreal.MaterialInterface):
        return {"success": False, "error": "Parent material not found: " + parent}
    if unreal.EditorAssetLibrary.does_asset_exist(destination):
        return {"success": False, "error": "Asset already exists: " + destination}
    # The factory's InitialParent is not exposed to Python; parent it after.
    instance = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, folder, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew(),
    )
    if instance is None:
        return {"success": False, "error": "Could not create material instance: " + destination}
    unreal.MaterialEditingLibrary.set_material_instance_parent(instance, material)
    unreal.EditorAssetLibrary.save_loaded_asset(instance)
    return {"instance_path": instance.get_path_name(), "parent_path": material.get_path_name(),
            "class_name": instance.get_class().get_name()}

print(json.dumps(create(**args)))
"""

# Expects ``args`` = {"actor_path", "material_path", "slot_index"}. Assigns the
# material as the Details panel does: inside an undoable transaction, after
# modify() records the component, which also marks the package that holds it
# (the level, or the actor's own package in a World Partition level) dirty, so
# saving keeps the change. MeshComponent.set_material alone, which the tool
# called before, records and dirties nothing: on UE 5.7.4 "save dirty
# packages" saved nothing, reloading the level reverted the material and Undo
# did not undo it. The actor must be in the level being edited (actor_at),
# and the material is loaded only when the asset registry lists it as
# something other than a level, so a map path never loads that map. The slot
# is read back after the write.
ASSIGN_MATERIAL = ACTOR_HELPERS + """
def mesh_component(actor):
    root = actor.root_component
    if isinstance(root, unreal.MeshComponent):
        return root
    components = actor.get_components_by_class(unreal.MeshComponent)
    if len(components) != 1:
        raise ValueError("Actor must have a mesh root component or exactly one MeshComponent: "
                         + actor.get_path_name())
    return components[0]

def material_at(path):
    package = path.split(".", 1)[0]
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    if any(str(asset.asset_class_path.asset_name) == "World"
           for asset in registry.get_assets_by_package_name(package)):
        return None
    material = unreal.load_asset(path)
    return material if isinstance(material, unreal.MaterialInterface) else None

def path_of(material):
    return material.get_path_name() if material is not None else None

def assign(actor_path, material_path, slot_index):
    actor = actor_at(actor_path)
    material = material_at(material_path)
    if material is None:
        return {"success": False, "error_type": "NotFoundError",
                "error": "No material at " + repr(material_path) + ": pass a Material or Material Instance "
                         "asset path such as /Engine/BasicShapes/BasicShapeMaterial"}
    component = mesh_component(actor)
    slots = component.get_num_materials()
    if not 0 <= slot_index < slots:
        return {"success": False, "error_type": "ValidationError",
                "error": ("slot_index " + str(slot_index) + " is out of range: " + component.get_name()
                          + " has " + str(slots) + " material slot(s)")}
    previous = path_of(component.get_material(slot_index))
    with unreal.ScopedEditorTransaction("simul: assign material"):
        component.modify()
        component.set_material(slot_index, material)
    applied = path_of(component.get_material(slot_index))
    result = {"actor_path": actor.get_path_name(), "component_path": component.get_path_name(),
              "material_path": material.get_path_name(), "slot_index": slot_index,
              "previous_material_path": previous}
    if applied != material.get_path_name():
        result.update(success=False, error_type="MaterialNotApplied",
                      error="Slot " + str(slot_index) + " of " + component.get_name() + " reads back "
                            + json.dumps(applied) + " after the assignment")
    return result

print(json.dumps(assign(**args)))
"""

# Defines material_parameters(material): every scalar, vector and texture
# parameter Unreal lists for a Material or Material Instance, by kind, as
# (name, association, index, value). The association is "global" for the
# material's own parameter, "layer" or "blend" for one inside its Material
# Layers, with the layer or blend index (-1 for a global parameter); the value
# is what the material renders with for that parameter. MaterialEditingLibrary
# lists a Material Layer's parameters with the material's own by name only,
# and reads and writes by name, which addresses the global parameter (with an
# association argument, layer 0, or a blend at index -1, which fails
# check(NewIndex > 0) in RemapLayerIndex for an instance of a layered material
# and took the editor down; observed on UE 5.7.4). So a transient
# Material Instance Dynamic parented to the material copies its parameters:
# UMaterialInstance::CopyMaterialInstanceParameters walks the parameter list
# MaterialEditingLibrary names from (GetAllParameterInfoOfType, which the
# instance shares with its parent) and stores each with its full
# FMaterialParameterInfo and the material's value for it, its overrides
# included, in the instance's own parameter arrays. A dynamic instance takes
# only a Material or a Material Instance Constant as its parent
# (UMaterialInstance::SetParentInternal leaves it without one otherwise, and it
# would list the default material's parameters), so the copy of a Material
# Instance Dynamic is parented to that one's parent, whose parameters it lists,
# or without one to the default material Unreal renders it with. Nothing is
# written to the material, and the copy works the same during Play In Editor.
MATERIAL_PARAMETERS = """
ASSOCIATIONS = {"GLOBAL_PARAMETER": "global", "LAYER_PARAMETER": "layer", "BLEND_PARAMETER": "blend"}
PARAMETER_VALUES = {"scalar": "scalar_parameter_values", "vector": "vector_parameter_values",
                    "texture": "texture_parameter_values"}

def parameter_key(entry):
    info = entry.parameter_info
    return str(info.name), ASSOCIATIONS[info.association.name], info.index

def material_parameters(material):
    parent = material
    if isinstance(material, unreal.MaterialInstance) and not isinstance(material, unreal.MaterialInstanceConstant):
        parent = material.get_editor_property("parent")
        if parent is None:
            parent = material.get_base_material()
    copy = unreal.MaterialLibrary.create_dynamic_material_instance(
        None, parent, creation_flags=unreal.MIDCreationFlags.TRANSIENT)
    copy.copy_material_instance_parameters(material, False)
    return {kind: [parameter_key(entry) + (entry.parameter_value,) for entry in copy.get_editor_property(values)]
            for kind, values in PARAMETER_VALUES.items()}
"""

# Expects ``args`` = {"path"}: a Material or Material Instance asset. Reads its
# scalar, vector and texture parameters through editor scripting: Remote
# Control's /remote/object/describe answered with the asset's object properties
# instead, and no parent (observed on UE 5.7.4). Each parameter is reported
# with its association and layer or blend index (MATERIAL_PARAMETERS), so a
# name the material and its Material Layers share is reported once for each,
# with that layer's value. A Material reports each parameter's default; an
# instance reports the value it renders with, its own override or what it
# inherits, and whether it overrides it, as the Material Instance editor shows.
# Overrides recorded for a name that is no longer a parameter of the material
# are not reported, as the editor does not show them. The blend mode is an
# instance's override, else its parent's, and opaque for an instance without a
# parent (as UMaterialInstance::UpdateOverridableBaseProperties sets it); the
# domain is the base material's. Loads through unreal.load_asset, so it answers
# the same during Play In Editor.
MATERIAL_INFO = """
import json, unreal
""" + MATERIAL_PARAMETERS + """
# The material's own parameters first, then each Material Layer's, then each blend's.
ORDER = {"global": 0, "layer": 1, "blend": 2}

def path_of(asset):
    return asset.get_path_name() if asset is not None else None

def plain(value):
    if isinstance(value, unreal.LinearColor):
        return [value.r, value.g, value.b, value.a]
    if isinstance(value, unreal.Object) or value is None:
        return path_of(value)
    return value

def overridden(instance, kind):
    # Parameter names ignore case, as Unreal compares them.
    return {(name.lower(), association, index) for name, association, index
            in map(parameter_key, instance.get_editor_property(PARAMETER_VALUES[kind]))}

def blend_mode(material):
    while isinstance(material, unreal.MaterialInstance):
        parent = material.get_editor_property("parent")
        if parent is None:
            # What Unreal renders an instance without a parent with, override or not.
            return "BLEND_OPAQUE"
        overrides = material.get_editor_property("base_property_overrides")
        if overrides.get_editor_property("override_blend_mode"):
            return overrides.get_editor_property("blend_mode").name
        material = parent
    return material.get_editor_property("blend_mode").name

def info(path):
    material = unreal.load_asset(path)
    if material is None:
        return {"success": False, "error_type": "NotFoundError",
                "error": "No material at " + repr(path) + ": pass a Material or Material Instance asset "
                         "path such as /Engine/BasicShapes/BasicShapeMaterial"}
    if not isinstance(material, unreal.MaterialInterface):
        return {"success": False, "error_type": "InvalidAssetType",
                "error": path + " is a " + material.get_class().get_name()
                         + ", not a Material or Material Instance"}
    instance = isinstance(material, unreal.MaterialInstance)
    found = material_parameters(material)
    parameters = []
    for kind in ("scalar", "vector", "texture"):
        overrides = overridden(material, kind) if instance else set()
        for name, association, index, value in sorted(
                found[kind], key=lambda parameter: (ORDER[parameter[1]], parameter[2], parameter[0].lower())):
            parameters.append({
                "name": name, "param_type": kind, "association": association,
                "index": None if association == "global" else index, "value": plain(value),
                "overridden": (name.lower(), association, index) in overrides if instance else None})
    base = material.get_base_material()
    return {"material_path": material.get_path_name(), "class_name": material.get_class().get_name(),
            "parent_path": path_of(material.get_editor_property("parent")) if instance else None,
            "base_material_path": path_of(base),
            "material_domain": base.get_editor_property("material_domain").name if base is not None else None,
            "blend_mode": blend_mode(material), "parameters": parameters}

print(json.dumps(info(**args)))
"""

# Expects ``args`` = {"actor_path", "values": {editor_property: value}}, with
# ``light_color`` as [r, g, b, a] bytes. Writes through set_editor_property
# (the edit path the Details panel uses): Remote Control property writes and
# the Set* functions silently no-op on a Stationary light's attenuation radius.
# Reads every value back so a write the engine refused is reported.
SET_LIGHT_PARAMS = ACTOR_HELPERS + """
component = actor_component(actor_at(args["actor_path"]), unreal.LightComponent)

def read(name):
    value = component.get_editor_property(name)
    if isinstance(value, unreal.Color):
        return [value.r, value.g, value.b, value.a]
    return value

# Fail on a property this light type lacks before anything is written.
for name in args["values"]:
    read(name)

with unreal.ScopedEditorTransaction("simul: set light params"):
    for name, value in args["values"].items():
        if name == "light_color":
            value = unreal.Color(r=value[0], g=value[1], b=value[2], a=value[3])
        component.set_editor_property(name, value)

applied = {name: read(name) for name in args["values"]}
rejected = sorted(
    name for name, wanted in args["values"].items()
    if (abs(applied[name] - wanted) > 1e-3 * max(1.0, abs(wanted))
        if isinstance(wanted, float) else applied[name] != wanted)
)
result = {"component_path": component.get_path_name(), "applied": applied,
          "params_set": len(args["values"]) - len(rejected)}
if rejected:
    result.update(success=False, error="Unreal did not apply: " + ", ".join(rejected))
print(json.dumps(result))
"""

#: ``error_type`` of a physics edit refused because the body cannot take it:
#: ``details.problems`` lists each reason as {"code", "message"}.
INCOMPATIBLE_BODY = "IncompatibleBody"

# Shared by the physics and collision scripts below. Each edits one primitive
# component (the actor's root, else its only one, as actor_component picks).
# An actor in the editor world is edited the way the Details panel does it:
# inside one ScopedEditorTransaction, after modify(), so the edit is one Undo
# step and marks the map dirty, and Save All writes it. The runtime setters
# alone (set_simulate_physics and the like) change the live body but neither
# record an undo step nor dirty the map, so the edit was lost on exit (observed
# on UE 5.7.4); after modify() they are recorded, Undo and Redo restore them,
# the attachment a simulating body drops included, and they survive a save and
# reload (UE 5.7.4). Cancelling a transaction only drops its record and keeps
# what was changed (UTransBuffer::Cancel), so a script that fails after
# writing restores the values itself before it cancels. An actor in a running
# simulation's world (UEDPIE_<n>_<map>) is the play copy: Unreal records no
# undo for it and saves nothing of it, so it is edited directly and the reply
# says so (world "simulation").
#
# Unreal accepts simulate_physics on any body and is_simulating_physics()
# reads True for one that never moves (UE 5.7.4): a Static or Stationary body
# (no mass, stays put), one whose collision has no physics (NoCollision or
# QueryOnly such as OverlapAll: "Invalid Simulate Options ... Collision Enabled
# is incompatible", and the body falls through the floor) and one whose mesh
# has no simple collision shape (falls through). physics_problems names them
# before anything is written. A mesh that uses its render triangles as simple
# collision (CTF_USE_COMPLEX_AS_SIMPLE) cannot simulate at all, though it can
# be a kinematic collider. StaticMeshEditorSubsystem reports -1 for every mesh
# while Play In Editor runs, so the mesh checks are skipped then.
PHYSICS_HELPERS = ACTOR_HELPERS + f"_INCOMPATIBLE_BODY = {INCOMPATIBLE_BODY!r}\n" + """
import re

PHYSICS_COLLISION = ("QUERY_AND_PHYSICS", "PHYSICS_ONLY")

def in_simulation(actor):
    world = actor.get_world()
    package = world.get_outermost().get_name().rpartition("/")[2] if world is not None else ""
    return re.match("UEDPIE_[0-9]+_", package) is not None

def physics_component(actor):
    return actor_component(actor, unreal.PrimitiveComponent)

def physics_mode(component):
    if component.get_collision_enabled().name not in PHYSICS_COLLISION:
        return "none"
    if component.mobility.name != "MOVABLE":
        return "static"
    body = component.get_editor_property("body_instance")
    if body.get_editor_property("simulate_physics") and component.is_simulating_physics():
        return "simulated"
    return "kinematic"

def physics_report(component):
    body = component.get_editor_property("body_instance")
    parent = component.get_attach_parent()
    override = bool(body.get_editor_property("override_mass"))
    return {
        "component_path": component.get_path_name(),
        "physics_mode": physics_mode(component),
        "simulate_physics": bool(body.get_editor_property("simulate_physics")),
        "mobility": component.mobility.name.title(),
        "collision_mode": component.get_collision_enabled().name,
        "collision_preset": str(component.get_collision_profile_name()),
        "mass_kg": component.get_mass(),
        "mass_override_kg": body.get_editor_property("mass_in_kg_override") if override else None,
        "gravity_enabled": bool(body.get_editor_property("enable_gravity")),
        "linear_damping": body.get_editor_property("linear_damping"),
        "angular_damping": body.get_editor_property("angular_damping"),
        "attach_parent": parent.get_path_name() if parent is not None else None,
    }

def mesh_collision(mesh):
    tools = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    simple = tools.get_simple_collision_count(mesh)
    convex = tools.get_convex_collision_count(mesh)
    if simple < 0 or convex < 0:
        return None
    return simple, convex, tools.get_collision_complexity(mesh).name

def physics_problems(component, simulate, make_movable, allow_detach):
    problems = []
    body = "a simulated" if simulate else "a kinematic"
    mobility = component.mobility.name
    if mobility != "MOVABLE" and not make_movable:
        problems.append({"code": "NOT_MOVABLE", "message": (
            "its mobility is " + mobility.title() + ", and Unreal moves and simulates only Movable "
            "bodies: pass make_movable true (the default) to make it Movable")})
    collision = component.get_collision_enabled().name
    if collision not in PHYSICS_COLLISION:
        problems.append({"code": "NO_PHYSICS_COLLISION", "message": (
            "its collision is " + collision + " (preset " + str(component.get_collision_profile_name())
            + "), and " + body + " body needs physics collision (QUERY_AND_PHYSICS or PHYSICS_ONLY): "
            "call set_unreal_collision with collision_preset PhysicsActor, or with collision_enabled "
            "true")})
    if isinstance(component, unreal.StaticMeshComponent):
        mesh = component.get_editor_property("static_mesh")
        if mesh is None:
            problems.append({"code": "NO_GEOMETRY", "message": (
                "it has no static mesh, so Unreal gives it no physics body")})
        else:
            counts = mesh_collision(mesh)
            if counts is not None:
                simple, convex, complexity = counts
                if complexity == "CTF_USE_COMPLEX_AS_SIMPLE":
                    if simulate:
                        problems.append({"code": "COMPLEX_AS_SIMPLE", "message": (
                            "its mesh " + mesh.get_path_name() + " uses its render triangles as simple "
                            "collision (Collision Complexity: Use Complex Collision As Simple), which "
                            "Unreal cannot simulate: set the mesh's Collision Complexity to Project "
                            "Default and give it simple collision, or make the body kinematic")})
                elif simple + convex == 0:
                    problems.append({"code": "NO_SIMPLE_COLLISION", "message": (
                        "its mesh " + mesh.get_path_name() + " has no simple collision shape, so " + body
                        + " body would pass through everything: add simple collision to the mesh "
                        "(Static Mesh editor, Collision menu)")})
    parent = component.get_attach_parent()
    if simulate and parent is not None and not allow_detach:
        problems.append({"code": "ATTACHED", "message": (
            "it is attached to " + parent.get_path_name() + ", and Unreal detaches a body from its "
            "parent when it starts simulating (keeping its world transform): pass allow_detach true to "
            "detach it in the same undoable step")})
    return problems

def refusal(actor, what, problems, component):
    return {"success": False, "error_type": _INCOMPATIBLE_BODY,
            "error": "Cannot " + what + " " + actor.get_actor_label() + ": "
                     + "; ".join(problem["message"] for problem in problems),
            "details": {"problems": problems, "component_path": component.get_path_name()}}

def pie_running():
    return bool(unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor())

def edit_note(simulation):
    if simulation:
        return ("This changed the running simulation's copy of the actor only: Unreal records no undo "
                "for it and it is gone when the simulation stops.")
    if pie_running():
        return ("Play In Editor is running: this changed the editor world, which the running "
                "simulation does not see; it takes effect the next time the simulation starts.")
    return None
"""

# Expects ``args`` = {"actor_path", "mode" ("simulate", "kinematic" or "off"),
# "make_movable", "allow_detach", "transaction"}. "simulate" makes the body
# simulate physics, "kinematic" keeps its physics collision without simulating
# (simulated bodies hit it; it moves only when its transform is set) and "off"
# stops it simulating, collision and mobility kept. A body that cannot take
# the mode is refused with nothing written (IncompatibleBody). A Static or
# Stationary body is made Movable when make_movable is true, and a simulated
# body attached to a parent is detached, as Unreal does, only with
# allow_detach. If the body still does not read back as asked, every value is
# restored and the transaction cancelled.
ENABLE_PHYSICS = PHYSICS_HELPERS + """
def apply(component, mode):
    if mode != "off" and component.mobility.name != "MOVABLE":
        component.set_mobility(unreal.ComponentMobility.MOVABLE)
    component.set_simulate_physics(mode == "simulate")

def unmet(component, mode):
    flag = bool(component.get_editor_property("body_instance").get_editor_property("simulate_physics"))
    if mode == "off":
        return None if not flag else "the body still simulates physics"
    if component.mobility.name != "MOVABLE":
        return "the body is still " + component.mobility.name.title()
    if mode == "kinematic":
        return None if not flag else "the body still simulates physics"
    if not (flag and component.is_simulating_physics()):
        return "Unreal did not start simulating the body"
    if component.get_mass() <= 0:
        return "Unreal gives the body no mass"
    return None

def restore(component, before):
    component.set_simulate_physics(before["simulate"])
    if component.mobility != before["mobility"]:
        component.set_mobility(before["mobility"])
    parent = before["parent"]
    if parent is not None and component.get_attach_parent() != parent:
        rule = unreal.AttachmentRule.KEEP_WORLD
        component.attach_to_component(parent, before["socket"], rule, rule, rule, False)

def enable():
    actor = actor_at(args["actor_path"], play_world=True)
    component = physics_component(actor)
    mode = args["mode"]
    if mode != "off":
        problems = physics_problems(component, mode == "simulate", args["make_movable"], args["allow_detach"])
        if problems:
            what = "simulate physics on" if mode == "simulate" else "make kinematic"
            return refusal(actor, what, problems, component)
    before = {"simulate": bool(component.get_editor_property("body_instance").get_editor_property("simulate_physics")),
              "mobility": component.mobility, "parent": component.get_attach_parent(),
              "socket": component.get_attach_socket_name()}
    simulation = in_simulation(actor)
    failure = None
    if simulation:
        apply(component, mode)
        failure = unmet(component, mode)
        if failure is not None:
            restore(component, before)
    else:
        with unreal.ScopedEditorTransaction(args["transaction"]) as transaction:
            component.modify()
            if before["parent"] is not None:
                before["parent"].modify()
            try:
                apply(component, mode)
                failure = unmet(component, mode)
            except Exception as exc:
                failure = str(exc) or type(exc).__name__
            if failure is not None:
                restore(component, before)
                transaction.cancel()
    if failure is not None:
        return {"success": False, "error_type": "PhysicsNotApplied",
                "error": failure + "; nothing was changed", "details": physics_report(component)}
    result = physics_report(component)
    result.update(actor_path=actor.get_path_name(), world="simulation" if simulation else "editor",
                  physics_enabled=result["physics_mode"] == "simulated",
                  mobility_changed_from=None if component.mobility == before["mobility"]
                  else before["mobility"].name.title(),
                  detached_from=None)
    warnings = []
    if before["parent"] is not None and component.get_attach_parent() is None:
        result["detached_from"] = before["parent"].get_path_name()
        warnings.append("Detached from " + result["detached_from"] + ", as Unreal does when a body starts "
                        "simulating; Undo reattaches it, turning physics off does not "
                        "(set_unreal_actor_parent does).")
    note = edit_note(simulation)
    if note:
        warnings.append(note)
    result["warning"] = " ".join(warnings) or None
    return result

print(json.dumps(enable()))
"""

# Expects ``args`` = {"actor_path", "values": {"mass", "linear_damping",
# "angular_damping", "enable_gravity"} (only those given), "transaction"}.
# Writes them with the body's setters in one undoable transaction and reads
# every value back (mass as the override stored, PhysicsParamsNotApplied for
# one that reads back different). With no values it only reads the body.
# Unreal gives a Static or Stationary body no mass (get_mass() reads 0) and
# never simulates it, so the reply warns that the values apply once the body
# is Movable.
SET_PHYSICS_PARAMS = PHYSICS_HELPERS + """
import struct

def stored(value):
    return struct.unpack("f", struct.pack("f", value))[0]

def write(component, values):
    if "mass" in values:
        component.set_mass_override_in_kg("None", values["mass"], True)
    if "linear_damping" in values:
        component.set_linear_damping(values["linear_damping"])
    if "angular_damping" in values:
        component.set_angular_damping(values["angular_damping"])
    if "enable_gravity" in values:
        component.set_enable_gravity(values["enable_gravity"])

def set_params():
    actor = actor_at(args["actor_path"], play_world=True)
    component = physics_component(actor)
    values = args["values"]
    simulation = in_simulation(actor)
    if values:
        if simulation:
            write(component, values)
        else:
            with unreal.ScopedEditorTransaction(args["transaction"]):
                component.modify()
                write(component, values)
    result = physics_report(component)
    read = {"mass": result["mass_override_kg"], "linear_damping": result["linear_damping"],
            "angular_damping": result["angular_damping"], "enable_gravity": result["gravity_enabled"]}
    applied = {name: read[name] for name in values}
    rejected = sorted(name for name, wanted in values.items()
                      if (read[name] != wanted if isinstance(wanted, bool)
                          else read[name] is None or abs(read[name] - stored(wanted)) > 1e-6 * max(1.0, abs(wanted))))
    result.update(actor_path=actor.get_path_name(), world="simulation" if simulation else "editor",
                  applied=applied, params_set=len(values) - len(rejected))
    warnings = []
    if result["mobility"] != "Movable":
        warnings.append("The body is " + result["mobility"] + ": Unreal gives such a body no mass (mass_kg "
                        "reads 0) and never simulates it. The values are stored and apply once it is "
                        "Movable (enable_unreal_physics makes it Movable).")
    elif result["physics_mode"] == "none":
        warnings.append("The body has no physics collision (" + result["collision_mode"] + "), so Unreal "
                        "gives it no physics mass and it collides with nothing.")
    note = edit_note(simulation)
    if note and values:
        warnings.append(note)
    result["warning"] = " ".join(warnings) or None
    if rejected:
        result.update(success=False, error_type="PhysicsParamsNotApplied",
                      error="Unreal did not apply: " + ", ".join(rejected))
    return result

print(json.dumps(set_params()))
"""

# Expects ``args`` = {"actor_path", "preset" (or None), "enabled" (True, False
# or None), "transaction"}. A preset is checked on a transient component
# first, since Unreal renames an unknown preset to Custom instead of failing,
# so an unknown one changes nothing. Setting collision_enabled on a static mesh
# component clears its Use Default Collision flag first, as setting a preset
# does in Unreal (UStaticMeshComponent::SetCollisionProfileName): with the flag
# set, Unreal puts the mesh's own collision back when the component is
# registered again, in the play copy when the simulation starts and after the
# map reloads, so a collision turned off came back as BlockAll (UE 5.7.4).
SET_COLLISION = PHYSICS_HELPERS + """
def known_preset(name):
    probe = unreal.new_object(unreal.StaticMeshComponent)
    probe.set_collision_profile_name(name)
    return str(probe.get_collision_profile_name()).lower() == name.lower()

def write(component, preset, enabled):
    if preset:
        component.set_collision_profile_name(preset)
    if enabled is not None:
        if (isinstance(component, unreal.StaticMeshComponent)
                and component.get_editor_property("use_default_collision")):
            component.set_editor_property("use_default_collision", False)
        component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS if enabled
                                        else unreal.CollisionEnabled.NO_COLLISION)

def set_collision():
    actor = actor_at(args["actor_path"], play_world=True)
    component = physics_component(actor)
    preset, enabled = args["preset"], args["enabled"]
    if preset and not known_preset(preset):
        return {"success": False, "error_type": "ValidationError",
                "error": "Unknown collision preset: " + preset + "; nothing was changed. Presets include "
                         "BlockAll, BlockAllDynamic, PhysicsActor, OverlapAll, NoCollision and Custom"}
    simulation = in_simulation(actor)
    if preset or enabled is not None:
        if simulation:
            write(component, preset, enabled)
        else:
            with unreal.ScopedEditorTransaction(args["transaction"]):
                component.modify()
                write(component, preset, enabled)
    result = physics_report(component)
    result.update(actor_path=actor.get_path_name(), world="simulation" if simulation else "editor",
                  collision_enabled=result["collision_mode"] != "NO_COLLISION",
                  use_default_collision=bool(component.get_editor_property("use_default_collision"))
                  if isinstance(component, unreal.StaticMeshComponent) else None)
    warnings = []
    if result["simulate_physics"] and result["collision_mode"] not in PHYSICS_COLLISION:
        effect = ("does not move at all" if result["collision_mode"] == "NO_COLLISION"
                  else "falls through everything")
        warnings.append("The body simulates physics but now has no physics collision ("
                        + result["collision_mode"] + "): in the simulation it " + effect
                        + ", and Unreal logs Invalid Simulate Options.")
    note = edit_note(simulation)
    if note and (preset or enabled is not None):
        warnings.append(note)
    result["warning"] = " ".join(warnings) or None
    return result

print(json.dumps(set_collision()))
"""

# Expects ``args`` = {"path", "values"}, where ``values`` maps "scalar",
# "vector" and "texture" to {parameter name: value}: a float, [r, g, b, a]
# floats, a texture asset path. Writes the Material Instance Constant through
# MaterialEditingLibrary, the calls its editor makes, in one undoable
# transaction. A Remote Control write of the instance's parameter arrays
# answered OK and changed nothing, and the setters return False whether or not
# they applied the value (observed on UE 5.7.4, in and out of Play In Editor),
# so like SET_LIGHT_PARAMS this reads every value back and fails the call when
# one reads back different (MaterialParamsNotApplied); the values that did
# apply stay written and are reported in applied, as lights do. A setter also
# adds an override for a name the material has no parameter of, so before
# anything is written the names are matched against the material's global
# parameters (MATERIAL_PARAMETERS; ignoring case, as Unreal's names do) and the
# textures are loaded: a name the material lacks or a texture that does not
# load refuses the whole request (ValidationError, nothing written, each reason
# in not_applied). So does a name only the material's Material Layers have:
# the setters write by name, and the global override they would add reads back
# as set while the layer renders as it was (observed on UE 5.7.4). A name both
# the material and a layer have sets the material's own parameter. Assets load
# through unreal.load_asset: EditorAssetLibrary loads nothing during PIE. The
# instance is left unsaved, as an edit in its editor is. An instance outside
# /Game (engine or plugin content, which every project using that engine or
# plugin shares, such as /Engine/BasicShapes/BasicShapeMaterial_Inst) is
# refused as a SandboxError unless ``args`` has "allow_engine_content" true:
# editing one left the shared engine asset dirty (observed on UE 5.7.4).
SET_MATERIAL_PARAMS = """
import json, struct, unreal
""" + MATERIAL_PARAMETERS + """
MEL = unreal.MaterialEditingLibrary
LISTED = 20

def stored(value):
    # Unreal keeps material scalars and colors as 32-bit floats.
    return struct.unpack("f", struct.pack("f", value))[0]

def expected(kind, value):
    if kind == "scalar":
        return stored(value)
    if kind == "vector":
        return [stored(channel) for channel in value]
    return value.get_path_name()

def write(instance, kind, name, value):
    if kind == "scalar":
        MEL.set_material_instance_scalar_parameter_value(instance, name, value)
    elif kind == "vector":
        MEL.set_material_instance_vector_parameter_value(instance, name, unreal.LinearColor(*value))
    else:
        MEL.set_material_instance_texture_parameter_value(instance, name, value)

def read(instance, kind, name):
    if kind == "scalar":
        return MEL.get_material_instance_scalar_parameter_value(instance, name)
    if kind == "vector":
        color = MEL.get_material_instance_vector_parameter_value(instance, name)
        return [color.r, color.g, color.b, color.a]
    texture = MEL.get_material_instance_texture_parameter_value(instance, name)
    return texture.get_path_name() if texture is not None else None

def missing(kind, name, names, layered):
    # names: the material's global parameters of this kind; layered: whether its layers have any.
    scope = "global " + kind if layered else kind
    if not names:
        return "the material has no " + scope + " parameters"
    listed = ", ".join(names[:LISTED]) + (", ..." if len(names) > LISTED else "")
    return ("the material has no " + kind + " parameter named " + repr(name)
            + " (its " + scope + " parameters: " + listed + ")")

def in_layers(kind, name, places):
    # places: (association, index) of each layer and blend parameter by that name.
    where = ", ".join(association + " " + str(index) for association, index
                      in sorted(places, key=lambda place: (place[0] != "layer", place[1])))
    return ("only the material's Material Layers have a " + kind + " parameter named " + repr(name)
            + " (" + where + "), and this tool sets global parameters only: a global override by that "
            "name would not change them")

def described(reasons):
    return "; ".join(name + " (" + kind + "): " + reason
                     for kind, entries in reasons.items() for name, reason in entries.items())

def set_params(path, values, allow_engine_content=False):
    instance = unreal.load_asset(path)
    if instance is None:
        return {"success": False, "error_type": "NotFoundError",
                "error": "No material instance at " + repr(path) + ": pass a Material Instance "
                         "Constant asset path such as /Game/Materials/MI_Wood"}
    if not isinstance(instance, unreal.MaterialInstanceConstant):
        error = path + " is a " + instance.get_class().get_name() + ", not a Material Instance Constant"
        if isinstance(instance, unreal.Material):
            error += "; create an instance of it with create_unreal_material_instance and set that"
        return {"success": False, "error_type": "InvalidAssetType", "error": error}
    if not allow_engine_content and not instance.get_path_name().startswith("/Game/"):
        return {"success": False, "error_type": "SandboxError",
                "error": ("Nothing was written to " + instance.get_path_name() + ": it is engine or plugin "
                          "content, outside the project's /Game folder, and every project using that engine "
                          "or plugin shares it"),
                "details": {"content_path": instance.get_path_name(), "allowed_root": "/Game",
                            "hint": "Create an instance of it under /Game with create_unreal_material_instance "
                                    "and set that, or pass allow_engine_content=true to edit the shared asset"}}
    found = material_parameters(instance)
    pending, refused = [], {}
    for kind, requested in values.items():
        # The material's own parameters, and where its layers have each name; names ignore case.
        by_key, layers = {}, {}
        for name, association, index, _ in found[kind]:
            if association == "global":
                by_key.setdefault(name.lower(), name)
            else:
                layers.setdefault(name.lower(), []).append((association, index))
        names = sorted(by_key.values(), key=str.lower)
        for name, value in requested.items():
            parameter = by_key.get(name.lower())
            if parameter is None:
                places = layers.get(name.lower())
                refused.setdefault(kind, {})[name] = (
                    in_layers(kind, name, places) if places else missing(kind, name, names, bool(layers)))
                continue
            if kind == "texture":
                texture = unreal.load_asset(value)
                if not isinstance(texture, unreal.Texture):
                    refused.setdefault(kind, {})[parameter] = "no texture at " + repr(value)
                    continue
                value = texture
            pending.append((kind, parameter, value))
    material_path = instance.get_path_name()
    if refused:
        return {"success": False, "error_type": "ValidationError",
                "error": "Nothing was written to " + material_path + ": " + described(refused),
                "material_path": material_path, "params_set": 0, "applied": {}, "not_applied": refused}
    with unreal.ScopedEditorTransaction("simul: set material params"):
        instance.modify()
        for kind, name, value in pending:
            write(instance, kind, name, value)
    MEL.update_material_instance(instance)
    applied, not_applied = {}, {}
    for kind, name, value in pending:
        value_read = read(instance, kind, name)
        if value_read == expected(kind, value):
            applied.setdefault(kind, {})[name] = value_read
        else:
            not_applied.setdefault(kind, {})[name] = (
                "Unreal read back " + json.dumps(value_read) + " after the write")
    result = {"material_path": material_path,
              "params_set": sum(len(entries) for entries in applied.values()),
              "applied": applied, "not_applied": not_applied}
    if not_applied:
        error = "Unreal did not apply every parameter to " + material_path + ": " + described(not_applied)
        if applied:
            error += ". The parameters in applied were set and stay set"
        result.update(success=False, error_type="MaterialParamsNotApplied", error=error)
    return result

print(json.dumps(set_params(**args)))
"""

# Expects ``args`` = {"pieces": [{"label", "mesh", "location", "scale",
# "movable"}], "folder", "tags", "transaction"}. Spawns one StaticMeshActor per
# piece, in order, inside one undoable transaction, and files them all in the
# outliner folder only once every piece has spawned: Unreal creates a folder
# the first time an actor is filed in it and keeps it, empty, when the actors
# go, and its Python API has no call that deletes a folder. When anything
# fails, rollback runs to the end whatever it meets: every actor spawned is
# destroyed (one Unreal refuses, or that raises, is reported and the rest are
# still destroyed) and the transaction is cancelled, so the level is left as it
# was, and the script prints a SpawnError. While PIE runs, when Unreal spawns
# nothing, it prints the PIE refusal instead.
PLACE_BASIC_SHAPES = EDITOR_WORLD + """
subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

def remove(actors):
    kept = []
    for actor in actors:
        try:
            path = actor.get_path_name()
        except Exception:
            path = "<actor that cannot be named>"
        try:
            if not subsystem.destroy_actor(actor):
                kept.append(path)
        except Exception:
            kept.append(path)
    return kept

def place():
    if pie_active():
        return pie_refusal("spawn actors")
    meshes, spawned, failure, filing, kept = {}, [], "interrupted", False, []
    with unreal.ScopedEditorTransaction(args["transaction"]) as transaction:
        try:
            for piece in args["pieces"]:
                mesh = meshes.get(piece["mesh"])
                if mesh is None:
                    mesh = unreal.load_asset(piece["mesh"])
                    if not isinstance(mesh, unreal.StaticMesh):
                        raise ValueError("no static mesh at " + piece["mesh"])
                    meshes[piece["mesh"]] = mesh
                actor = subsystem.spawn_actor_from_class(
                    unreal.StaticMeshActor, unreal.Vector(*piece["location"]), unreal.Rotator(0.0, 0.0, 0.0))
                if actor is None:
                    raise RuntimeError("Unreal spawned no actor for " + piece["label"])
                spawned.append(actor)
                component = actor.static_mesh_component
                component.set_mobility(unreal.ComponentMobility.MOVABLE if piece["movable"]
                                       else unreal.ComponentMobility.STATIC)
                component.set_static_mesh(mesh)
                actor.set_actor_scale3d(unreal.Vector(*piece["scale"]))
                actor.set_actor_label(piece["label"])
                actor.set_editor_property("tags", [unreal.Name(tag) for tag in args["tags"]])
            filing = True
            for actor in spawned:
                actor.set_folder_path(args["folder"])
            failure = None
        except Exception as exc:
            failure = str(exc) or type(exc).__name__
        finally:
            if failure is not None:
                try:
                    kept = remove(spawned)
                finally:
                    transaction.cancel()
    if failure is None:
        return {"actors_spawned": [actor.get_path_name() for actor in spawned]}
    error = failure
    if kept:
        error += ("; rollback could not remove " + str(len(kept)) + " of the " + str(len(spawned))
                  + " actors spawned: " + ", ".join(kept))
    else:
        error = "Spawned nothing: " + error
    if filing:
        error += ("; the outliner folder " + args["folder"] + " may remain empty, as Unreal's "
                  "Python API cannot delete outliner folders")
    result = {"success": False, "error_type": "SpawnError", "error": error}
    if kept:
        result["details"] = {"actors_left": kept}
    return result

print(json.dumps(place()))
"""

# Expects ``args`` = {"analyses", "actor_filter", "criteria", "listed_limit"}.
# Reads every actor loaded in the editor world whose label or object path
# contains actor_filter, ignoring case. Only primitive components with
# collision enabled count, and an actor's size is the bounding box of those
# components. Static mesh collision comes from the mesh asset
# (StaticMeshEditorSubsystem): simple shapes, convex hulls and whether the
# render triangles stand in for simple collision (CTF_USE_COMPLEX_AS_SIMPLE).
# See UnrealRuntimeSession.analyze_scene_for_robotics for the classification.
# While PIE runs the editor world is read (level_actors) and pie_active says so;
# StaticMeshEditorSubsystem then answers -1 and the default complexity for
# every mesh (observed on UE 5.7.4), so the mesh collision counts are null, and
# so is the list of actors missing simple collision, which those answers decide.
ROBOTICS_SCENE_ANALYSIS = EDITOR_WORLD + """
import math
analyses = set(args["analyses"])
criteria = args["criteria"]
needle = (args["actor_filter"] or "").lower()
mesh_tools = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
mesh_collision = {}
pie = pie_active()

def collision_of(mesh):
    key = mesh.get_path_name()
    if key not in mesh_collision:
        mesh_collision[key] = (mesh_tools.get_simple_collision_count(mesh),
                               mesh_tools.get_convex_collision_count(mesh),
                               mesh_tools.get_collision_complexity(mesh).name)
    return mesh_collision[key]

def entry(actor, size, **extra):
    return dict({"name": actor.get_actor_label(), "path": actor.get_path_name(),
                 "class_name": actor.get_class().get_name(),
                 "size": [round(value, 2) for value in size]}, **extra)

traversable, graspable = [], []
summary = {"total_actors": 0, "with_collision": 0, "physics_collision": 0,
           "static_mesh_components": 0, "simple_shapes": 0, "convex_hulls": 0,
           "complex_collision": 0, "missing_simple_collision": 0, "heightfields": 0,
           "by_profile": {}, "missing_simple_collision_actors": []}
for actor in level_actors():
    if needle and needle not in actor.get_actor_label().lower() and needle not in actor.get_path_name().lower():
        continue
    summary["total_actors"] += 1
    colliding = []
    for component in actor.get_components_by_class(unreal.PrimitiveComponent):
        state = component.get_collision_enabled().name
        if state != "NO_COLLISION":
            colliding.append((component, state in ("PHYSICS_ONLY", "QUERY_AND_PHYSICS")))
    if not colliding:
        continue
    physical = [component for component, physics in colliding if physics]
    summary["with_collision"] += 1
    if physical:
        summary["physics_collision"] += 1
    missing = False
    for component, physics in colliding:
        if "collision_complexity" not in analyses:
            break
        profile = str(component.get_collision_profile_name())
        summary["by_profile"][profile] = summary["by_profile"].get(profile, 0) + 1
        if component.get_class().get_name() == "LandscapeHeightfieldCollisionComponent":
            summary["heightfields"] += 1
        elif isinstance(component, unreal.StaticMeshComponent):
            mesh = component.get_editor_property("static_mesh")
            if mesh is None:
                continue
            summary["static_mesh_components"] += 1
            if pie:
                continue
            simple, convex, complexity = collision_of(mesh)
            summary["simple_shapes"] += simple
            summary["convex_hulls"] += convex
            if complexity == "CTF_USE_COMPLEX_AS_SIMPLE":
                summary["complex_collision"] += 1
            elif physics and simple + convex == 0:
                summary["missing_simple_collision"] += 1
                missing = True
    if missing and len(summary["missing_simple_collision_actors"]) < args["listed_limit"]:
        summary["missing_simple_collision_actors"].append(actor.get_path_name())
    if not physical:
        continue
    origin, extent = actor.get_actor_bounds(True)
    size = [2 * extent.x, 2 * extent.y, 2 * extent.z]
    if "traversability" in analyses:
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, actor.get_actor_up_vector().z))))
        area = size[0] * size[1] / 10000.0
        if (area >= criteria["surface_min_area_m2"]
                and size[2] <= criteria["surface_max_thickness_ratio"] * min(size[0], size[1])
                and tilt <= criteria["surface_max_tilt_degrees"]):
            traversable.append(entry(actor, size, top_z=round(origin.z + extent.z, 2),
                                     area_m2=round(area, 3), tilt_degrees=round(tilt, 2)))
    root = actor.root_component
    if ("graspability" in analyses and root is not None and root.mobility.name == "MOVABLE"
            and 0 < max(size) <= criteria["grasp_max_size_cm"]):
        mass = sum(component.get_mass() for component in physical)
        simulated = any(component.get_editor_property("body_instance").get_editor_property("simulate_physics")
                        for component in physical)
        graspable.append(entry(actor, size, simulate_physics=bool(simulated),
                               mass_kg=round(mass, 3) if mass > 0 else None))
if pie:
    for key in ("simple_shapes", "convex_hulls", "complex_collision", "missing_simple_collision",
                "missing_simple_collision_actors"):
        summary[key] = None
print(json.dumps({"traversable_surfaces": traversable, "graspable_objects": graspable,
                  "collision_summary": summary if "collision_complexity" in analyses else {},
                  "total_actors_analyzed": summary["total_actors"], "pie_active": pie}))
"""
