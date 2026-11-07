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

ACTOR_HELPERS = """
import json, unreal

def actor_at(path):
    actor = unreal.load_object(None, path)
    if actor is None or not isinstance(actor, unreal.Actor):
        raise ValueError("Actor not found: " + path)
    return actor

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

# Imported actors are found as the level actors that were not there before,
# a listing Unreal withholds while PIE runs, so the import is refused then.
USD_IMPORT = EDITOR_WORLD + """
import os
if not hasattr(unreal, "UsdStageImportFactory"):
    print(json.dumps({"success": False, "error_type": "PluginUnavailable",
        "error": "USD import requires the USDImporter plugin enabled at editor startup."}))
elif pie_active():
    print(json.dumps(pie_refusal("import USD into the level")))
else:
    if not os.path.isfile(args["path"]):
        raise ValueError("USD file not found on the editor host: " + args["path"])
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    before = {a.get_path_name() for a in subsystem.get_all_level_actors()}
    options = unreal.UsdStageImportOptions()
    for key, value in args["options"].items():
        options.set_editor_property(key, value)
    task = unreal.AssetImportTask()
    task.filename = args["path"]
    task.destination_path = args["destination"]
    task.automated = True
    task.save = True
    task.replace_existing = False
    task.factory = unreal.UsdStageImportFactory()
    task.options = options
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    assets = [str(p) for p in task.imported_object_paths]
    actors = [a.get_path_name() for a in subsystem.get_all_level_actors()
              if a.get_path_name() not in before]
    if not assets and not actors:
        raise RuntimeError("USD importer produced no assets or actors; check the editor log")
    print(json.dumps({"imported_assets": assets, "actor_paths": actors, "warnings": []}))
"""

# The export selects the actors and exports the editor world, neither of which
# Unreal allows while PIE runs, so the export is refused then.
USD_EXPORT = EDITOR_WORLD + """
import os
if not hasattr(unreal, "LevelExporterUSD"):
    print(json.dumps({"success": False, "error_type": "PluginUnavailable",
        "error": "USD export requires the USDImporter plugin enabled at editor startup."}))
elif pie_active():
    print(json.dumps(pie_refusal("export actors to USD")))
else:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = [actor_at(path) for path in args["actors"]]
    previous_selection = subsystem.get_selected_level_actors()
    # UE's native level exporter hands its task to Python through the CDO.
    options = unreal.get_default_object(unreal.LevelExporterUSDOptions)
    previous_inner = options.inner.copy()
    inner = previous_inner.copy()
    for key, value in args["options"].items():
        inner.set_editor_property(key, value)
    inner.selection_only = True
    # Secondary assets must stay under the authorized output directory too.
    inner.asset_folder = unreal.DirectoryPath(os.path.dirname(args["path"]))
    task = unreal.AssetExportTask()
    task.object = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    task.filename = args["path"]
    task.automated = True
    task.prompt = False
    task.selected = True
    task.replace_identical = True
    task.exporter = unreal.LevelExporterUSD()
    task.options = options
    try:
        options.inner = inner
        os.makedirs(os.path.dirname(args["path"]), exist_ok=True)
        subsystem.set_selected_level_actors(actors)
        if not unreal.Exporter.run_asset_export_task(task):
            raise RuntimeError("USD export failed: " + "; ".join(str(e) for e in task.errors))
        size = os.path.getsize(args["path"])
        if size == 0:
            raise RuntimeError("USD export produced an empty file")
        print(json.dumps({"output_path": args["path"], "actors_exported": len(actors),
                          "file_size_bytes": size}))
    finally:
        subsystem.set_selected_level_actors(previous_selection)
        options.inner = previous_inner
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
FOCUS_ACTOR = ACTOR_HELPERS + EDITOR_WORLD + """
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
SPAWN_ACTOR = ACTOR_HELPERS + EDITOR_WORLD + """
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

# world_partition_state(loaded): whether the editor world is a World Partition
# map and, if so, how many of its actors are not loaded in the editor, after
# EDITOR_WORLD. A World Partition map keeps an actor descriptor for every
# saved actor and loads only the actors of the regions the editor has loaded,
# so a map opened normally may have most of them unloaded, and Unreal's level
# listing (get_all_level_actors) leaves those out. Descriptors whose actor path
# is not among ``loaded`` (paths of level actors; None reads them) are the
# unloaded ones. WorldPartitionBlueprintLibrary returns no descriptors (None)
# outside a World Partition map, and finds no world while PIE runs (it asks
# UnrealEditorSubsystem.get_editor_world, which answers None then), so during
# PIE every field is None (observed on UE 5.7.4).
WORLD_PARTITION_HELPERS = """
def world_partition_state(loaded=None):
    if pie_active():
        return {"world_partition": None, "actor_descriptors": None, "unloaded_actors": None}
    descriptors = unreal.WorldPartitionBlueprintLibrary.get_actor_descs()
    if descriptors is None:
        return {"world_partition": False, "actor_descriptors": None, "unloaded_actors": None}
    if loaded is None:
        loaded = {actor.get_path_name() for actor in level_actors()}
    unloaded = sum(1 for descriptor in descriptors if str(descriptor.actor_path) not in loaded)
    return {"world_partition": True, "actor_descriptors": len(descriptors), "unloaded_actors": unloaded}
"""

#: Most child actor paths a listing entry names; its ``child_count`` counts
#: them all, and ``get_unreal_actor_info`` names more.
LISTED_CHILDREN = 20

#: Most child actor paths ``get_actor_info`` names; ``child_count`` counts them all.
INFO_CHILDREN = 200

# Helpers of the paged scene reads, after ACTOR_HELPERS and EDITOR_WORLD.
#
# hierarchy(actor, listed): the attachment hierarchy as Unreal keeps it. The
# parent is the actor whose component the actor's root component is attached
# to (Actor.get_attach_parent_actor), or None; the children are the actors
# attached to any of its components (Actor.get_attached_actors, direct ones
# only), the first ``listed`` of them by path, with their count.
#
# class_matcher(class_filter, include_subclasses, actors): a test of a UClass,
# and a warning or None. A class matches when its name (StaticMeshActor,
# BP_Door_C) or path (/Script/Engine.StaticMeshActor) is the filter. With
# include_subclasses a class also matches when it derives from the filter's
# class (MathLibrary.class_is_child_of, so a Blueprint class derived from
# PointLight matches Light). The filter's class is a /Script/ or Blueprint
# class path, a Blueprint asset path, the name of a native class
# (getattr(unreal, name)), or the name of the class of one of ``actors``, which
# is how a Blueprint class name resolves; a filter that resolves to no class
# matches by name only and comes with a warning.
#
# page(items, offset, limit, budget, render): the entries ``render`` makes of
# items[offset:], stopping before the limit-th entry or before the entry that
# would take the entries' JSON past ``budget`` characters (the first entry
# always fits, so every page moves on). Returns the entries, the offset of the
# next page (None when none is left) and what ended the page: "max_results",
# "response_budget" or None.
SCENE_READ_HELPERS = """
def hierarchy(actor, listed):
    parent = actor.get_attach_parent_actor()
    children = list(actor.get_attached_actors())
    return {"parent": parent.get_path_name() if parent is not None else None,
            "children": [child.get_path_name() for child in children[:listed]],
            "child_count": len(children)}

def resolve_class(name, actors):
    if "/" in name:
        found = unreal.load_class(None, name)
        if found is None and not name.endswith("_C"):
            asset = name if "." in name.rpartition("/")[2] else name + "." + name.rpartition("/")[2]
            found = unreal.load_class(None, asset + "_C")
        return found
    native = getattr(unreal, name, None)
    if isinstance(native, type) and hasattr(native, "static_class"):
        found = native.static_class()
        if isinstance(found, unreal.Class):
            return found
    for actor in actors:
        cls = actor.get_class()
        if cls.get_name() == name:
            return cls
    return None

def class_matcher(class_filter, include_subclasses, actors):
    if not class_filter:
        return (lambda cls: True), None
    base, warning = None, None
    if include_subclasses:
        base = resolve_class(class_filter, actors)
        if base is None:
            warning = ("No class named " + repr(class_filter) + " was found, so only actors of a class "
                       "with that name or path match. Pass a class path such as /Script/Engine.Light "
                       "or /Game/Blueprints/BP_Door.BP_Door_C")
    seen = {}

    def matches(cls):
        key = cls.get_path_name()
        if key not in seen:
            seen[key] = (class_filter in (cls.get_name(), key)
                         or (base is not None and unreal.MathLibrary.class_is_child_of(cls, base)))
        return seen[key]
    return matches, warning

def page(items, offset, limit, budget, render):
    entries, used, index, ended = [], 0, offset, None
    while index < len(items):
        if len(entries) == limit:
            ended = "max_results"
            break
        entry = render(items[index])
        size = len(json.dumps(entry)) + 2
        if entries and used + size > budget:
            ended = "response_budget"
            break
        entries.append(entry)
        used += size
        index += 1
    return entries, (index if index < len(items) else None), ended

def listing_entry(actor, listed):
    entry = {key: value for key, value in actor_info(actor).items() if key != "components"}
    entry.update(hierarchy(actor, listed))
    return entry
""" + WORLD_PARTITION_HELPERS

# Expects ``args`` = {"class_filter", "include_subclasses", "tag_filter",
# "label_filter", "offset", "max_results", "budget", "listed_children"}. Every
# level actor is tested (class_matcher; an actor tag equal to tag_filter, as
# Unreal's actor_has_tag compares; label_filter inside the actor label,
# ignoring case) and counted in ``total``; one page of the matches, from
# ``offset``, is rendered (page). While PIE runs the editor world is read
# (level_actors) and pie_active says so.
LIST_ACTORS = ACTOR_HELPERS + EDITOR_WORLD + SCENE_READ_HELPERS + """
actors = level_actors()
matches_class, warning = class_matcher(args["class_filter"], args["include_subclasses"], actors)
label = (args["label_filter"] or "").lower()
found = [actor for actor in actors
         if matches_class(actor.get_class())
         and (not args["tag_filter"] or actor.actor_has_tag(args["tag_filter"]))
         and (not label or label in actor.get_actor_label().lower())]
entries, next_offset, ended = page(found, args["offset"], args["max_results"], args["budget"],
                                   lambda actor: listing_entry(actor, args["listed_children"]))
print(json.dumps(dict({"actors": entries, "count": len(entries), "total": len(found), "offset": args["offset"],
                       "next_offset": next_offset, "page_ended_by": ended,
                       "warnings": [warning] if warning else [], "pie_active": pie_active()},
                      **world_partition_state({actor.get_path_name() for actor in actors}))))
"""

# Expects ``args`` = {"actor_path", "listed_children"}: actor_info with the
# attachment hierarchy (hierarchy) and pie_active.
ACTOR_INFO = ACTOR_HELPERS + EDITOR_WORLD + SCENE_READ_HELPERS + """
actor = actor_at(args["actor_path"])
print(json.dumps(dict(actor_info(actor), **hierarchy(actor, args["listed_children"]), pie_active=pie_active())))
"""

# Expects ``args`` = {"label", "match_mode", "offset", "max_results",
# "budget", "listed_children"}. Semantic labels are actor tags. Matching
# ignores case, as Unreal's tag comparison (FName) does: "exact" compares whole
# tags, "contains" looks for the label inside a tag and "regex" searches each
# tag for the regular expression. Every match is counted; one page of them,
# from ``offset``, is returned as listing entries (page). While PIE runs the
# editor world is searched (level_actors) and pie_active says so.
FIND_ACTORS_BY_TAG = ACTOR_HELPERS + EDITOR_WORLD + SCENE_READ_HELPERS + """
label, mode = args["label"], args["match_mode"]
if mode == "regex":
    pattern = re.compile(label, re.IGNORECASE)

def matches(tag):
    if mode == "regex":
        return pattern.search(tag) is not None
    if mode == "contains":
        return label.lower() in tag.lower()
    return tag.lower() == label.lower()

actors = level_actors()
found = [actor for actor in actors
         if any(matches(str(tag)) for tag in actor.get_editor_property("tags"))]
entries, next_offset, ended = page(found, args["offset"], args["max_results"], args["budget"],
                                   lambda actor: listing_entry(actor, args["listed_children"]))
print(json.dumps(dict({"actors": entries, "count": len(entries), "total_matches": len(found),
                       "offset": args["offset"], "next_offset": next_offset, "page_ended_by": ended,
                       "label_searched": label, "warnings": [], "pie_active": pie_active()},
                      **world_partition_state({actor.get_path_name() for actor in actors}))))
"""

# Expects ``args`` = {"root_path", "max_depth", "include_components",
# "class_filter", "include_subclasses", "offset", "max_results", "budget"}.
# The attachment hierarchy of the level actors: an actor's parent is the
# actor its root component is attached to (get_attach_parent_actor), and an
# actor attached to nothing, or to an actor the level does not list, is a top
# actor. The root is the level, whose children are the top actors, or the
# actor at root_path, whose children are the actors attached to it. With a
# class filter (class_matcher) an actor is kept when it matches or when one of
# its descendants does, so the path down to every match stays; a kept actor
# that does not match has matches_filter false. One page of the root's kept
# children, from ``offset``, is rendered with their descendants down to
# max_depth levels below the root (the root's children are level 1); an actor
# whose children are below that has children_truncated. When the first
# subtree of a page alone is over the budget, it is rendered shallower until it
# fits, and depth_limited names it. total_actors and total_depth count every
# kept actor below the root, whatever the page and max_depth. Components, with
# include_components, are each actor's components with their class and the
# component of the same actor they are attached to (None for the root and for
# components that are not scene components). While PIE runs the editor world is
# read (level_actors) and pie_active says so.
SCENE_GRAPH = ACTOR_HELPERS + EDITOR_WORLD + SCENE_READ_HELPERS + """
actors = level_actors()
listed = {actor.get_path_name() for actor in actors}
children, tops = {}, []
for actor in actors:
    parent = actor.get_attach_parent_actor()
    if parent is not None and parent.get_path_name() in listed:
        children.setdefault(parent.get_path_name(), []).append(actor)
    else:
        tops.append(actor)
matches_class, warning = class_matcher(args["class_filter"], args["include_subclasses"], actors)
filtering = bool(args["class_filter"])
keeps = {}

def kept(actor):
    path = actor.get_path_name()
    if path not in keeps:
        keeps[path] = True
        keeps[path] = (matches_class(actor.get_class())
                       or any([kept(child) for child in children.get(path, [])]))
    return keeps[path]

def kept_children(actor):
    return [child for child in children.get(actor.get_path_name(), []) if kept(child)]

def components_of(actor):
    root = actor.root_component
    listed_components = []
    for component in actor.get_components_by_class(unreal.ActorComponent):
        attach = component.get_attach_parent() if isinstance(component, unreal.SceneComponent) else None
        listed_components.append({
            "name": component.get_name(), "class_name": component.get_class().get_name(),
            "is_root": component == root,
            "attach_parent": attach.get_name() if attach is not None and component != root else None})
    return listed_components

def node(actor, depth, max_depth):
    below = kept_children(actor)
    entry = {"name": actor.get_actor_label(), "path": actor.get_path_name(),
             "class_name": actor.get_class().get_name(), "child_count": len(below), "children": []}
    if filtering:
        entry["matches_filter"] = matches_class(actor.get_class())
    if args["include_components"]:
        entry["components"] = components_of(actor)
    if depth < max_depth:
        entry["children"] = [node(child, depth + 1, max_depth) for child in below]
    elif below:
        entry["children_truncated"] = True
    return entry

if args["root_path"]:
    root_actor = actor_at(args["root_path"])
    top = kept_children(root_actor)
    root = node(root_actor, 0, 0)
    root.pop("children_truncated", None)
else:
    world = editor_world()
    top = [actor for actor in tops if kept(actor)]
    root = {"name": world.get_name() if world else "", "path": world.get_path_name() if world else "",
            "class_name": "World", "child_count": len(top), "children": []}

total_actors, total_depth, stack = 0, 0, [(actor, 1) for actor in top]
while stack:
    actor, depth = stack.pop()
    total_actors += 1
    total_depth = max(total_depth, depth)
    stack.extend((child, depth + 1) for child in kept_children(actor))

shallower = {}

def render(actor):
    depth = args["max_depth"]
    entry = node(actor, 1, depth)
    while depth > 1 and len(json.dumps(entry)) > args["budget"]:
        depth -= 1
        entry = node(actor, 1, depth)
        shallower[entry["path"]] = depth
    return entry

entries, next_offset, ended = page(top, args["offset"], args["max_results"], args["budget"], render)
root["children"] = entries

def count_nodes(entry):
    return 1 + sum(count_nodes(child) for child in entry["children"])

print(json.dumps(dict({"root": root, "total_actors": total_actors, "total_depth": total_depth,
                       "returned_actors": sum(count_nodes(entry) for entry in entries),
                       "count": len(entries), "total": len(top), "offset": args["offset"],
                       "next_offset": next_offset, "page_ended_by": ended,
                       "depth_limited": [{"path": entry["path"], "depth": shallower[entry["path"]]}
                                         for entry in entries if entry["path"] in shallower],
                       "warnings": [warning] if warning else [], "pie_active": pie_active()},
                      **world_partition_state(listed))))
"""

# Expects ``args`` = {"actor_path", "class_filter", "offset", "max_results",
# "budget"}. Every component of the actor (get_components_by_class
# ActorComponent, editor-only ones such as sprites included), or those whose
# class name or path is class_filter or derives from it (class_matcher with
# subclasses). Scene components report the component they are attached to
# (its name when it is the actor's own, its path when it belongs to another
# actor: the root of an attached actor), their world location, mobility and
# visibility; primitive components also their world bounds
# (SystemLibrary.get_component_bounds: the box's center and half-size and the
# bounding sphere's radius, in cm). One page, from ``offset`` (page).
ACTOR_COMPONENTS = ACTOR_HELPERS + EDITOR_WORLD + SCENE_READ_HELPERS + """
actor = actor_at(args["actor_path"])
root = actor.root_component
found = list(actor.get_components_by_class(unreal.ActorComponent))
if args["class_filter"]:
    matches_class, warning = class_matcher(args["class_filter"], True, [])
    found = [component for component in found if matches_class(component.get_class())]
else:
    warning = None

def vector(value):
    return [value.x, value.y, value.z]

def component_entry(component):
    entry = {"name": component.get_name(), "path": component.get_path_name(),
             "class_name": component.get_class().get_name(), "is_root": component == root,
             "is_scene_component": isinstance(component, unreal.SceneComponent),
             "editor_only": bool(component.get_editor_property("is_editor_only")),
             "attach_parent": None, "location": None, "mobility": None, "visible": None, "bounds": None}
    if isinstance(component, unreal.SceneComponent):
        attach = component.get_attach_parent()
        if attach is not None:
            entry["attach_parent"] = (attach.get_name() if attach.get_owner() == actor
                                      else attach.get_path_name())
        entry["location"] = vector(component.get_world_location())
        entry["mobility"] = component.mobility.name.title()
        entry["visible"] = bool(component.is_visible())
    if isinstance(component, unreal.PrimitiveComponent):
        origin, extent, radius = unreal.SystemLibrary.get_component_bounds(component)
        entry["bounds"] = {"origin": vector(origin), "extent": vector(extent), "sphere_radius": radius}
    return entry

entries, next_offset, ended = page(found, args["offset"], args["max_results"], args["budget"], component_entry)
print(json.dumps({"actor_path": actor.get_path_name(), "components": entries, "count": len(entries),
                  "total": len(found), "offset": args["offset"], "next_offset": next_offset,
                  "page_ended_by": ended, "warnings": [warning] if warning else [],
                  "pie_active": pie_active()}))
"""

# Scene digest in one round trip. Class keys are UClass path names
# (``/Script/Engine.StaticMeshActor``), the same strings Remote Control's
# ``/remote/object/describe`` reports as ``Class``.
SCENE_SUMMARY = EDITOR_WORLD + WORLD_PARTITION_HELPERS + """
world = editor_world()
counts = {}
total = 0
for actor in level_actors():
    name = actor.get_class().get_path_name()
    counts[name] = counts.get(name, 0) + 1
    total += 1
print(json.dumps({
    "map_path": world.get_path_name() if world else "",
    "total_actors": total,
    "actor_class_counts": counts,
    "pie_active": pie_active(),
    **world_partition_state(),
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
# instance is left unsaved, as an edit in its editor is.
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

def set_params(path, values):
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

# The Play In Editor state get_viewport_info adds to the level viewport camera:
# whether PIE runs, the player's view (PlayerCameraManager location, rotation
# and field of view, and the game viewport's size from WidgetLayoutLibrary)
# when a Play session has a player (Simulate In Editor has none), and the level
# viewport layout's panes.
VIEWPORT_STATE = """
import json, unreal
level = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
pie = bool(level.is_in_play_in_editor())
game = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world() if pie else None
manager = unreal.GameplayStatics.get_player_camera_manager(game, 0) if game is not None else None
view = None
if manager is not None:
    location, rotation = manager.get_camera_location(), manager.get_camera_rotation()
    size = unreal.WidgetLayoutLibrary.get_viewport_size(game)
    view = {"camera_location": [location.x, location.y, location.z],
            "camera_rotation": [rotation.pitch, rotation.yaw, rotation.roll],
            "fov": manager.get_fov_angle(), "viewport_size": [int(round(size.x)), int(round(size.y))]}
print(json.dumps({"pie_active": pie, "game_view": view,
                  "viewports": [str(key) for key in level.get_viewport_config_keys()],
                  "active_viewport": str(level.get_active_viewport_config_key())}))
"""

#: Most actor paths a World Partition load or unload names; its counts cover all.
WORLD_PARTITION_LISTED = 50

# Expects ``args`` = {"action", "bounds_min", "bounds_max", "max_actors",
# "listed"}: action "load" or "unload", the box as [x, y, z] corners in cm (both
# None for every actor descriptor of the map). Loads (or unloads) the actors of
# the World Partition descriptors whose editor bounds intersect the box,
# through WorldPartitionBlueprintLibrary.load_actors / unload_actors, the
# library's own loader: it loads them in the editor as the World Partition
# editor's Load does, at once, and leaves the map package clean (observed on UE
# 5.7.4). Unloading asks only for spatially loaded actors (the others are
# always loaded), and unload_actors only unloads actors that load_actors
# loaded, so an actor a loading region of the editor keeps loaded stays loaded
# and is reported in not_changed. Refused when more than max_actors would change (nothing is
# done), outside a World Partition map (InvalidState) and while PIE runs, when
# the library finds no editor world (PIEActive).
WORLD_PARTITION_LOAD = EDITOR_WORLD + WORLD_PARTITION_HELPERS + """
def change():
    if pie_active():
        return pie_refusal("load or unload World Partition actors")
    library = unreal.WorldPartitionBlueprintLibrary
    descriptors = library.get_actor_descs()
    if descriptors is None:
        return {"success": False, "error_type": "InvalidState",
                "error": "The editor world is not a World Partition map: every actor is loaded already"}
    if args["bounds_min"] is not None:
        box = unreal.Box(min=unreal.Vector(*args["bounds_min"]), max=unreal.Vector(*args["bounds_max"]))
        descriptors = library.get_intersecting_actor_descs(box) or []
    loaded = {actor.get_path_name() for actor in level_actors()}
    load = args["action"] == "load"
    # Always-loaded actors (not spatially loaded) cannot be unloaded.
    targets = [d for d in descriptors if (str(d.actor_path) in loaded) != load
               and (load or d.is_spatially_loaded)]
    if len(targets) > args["max_actors"]:
        return {"success": False, "error_type": "ValidationError",
                "error": (str(len(targets)) + " actors would " + args["action"] + ", more than max_actors ("
                          + str(args["max_actors"]) + "); nothing was " + args["action"] + "ed. Narrow "
                          "bounds_min and bounds_max, or raise max_actors"),
                "details": {"would_change": len(targets), "max_actors": args["max_actors"]}}
    guids = [d.guid for d in targets]
    if guids and load:
        unreal.WorldPartitionBlueprintLibrary.load_actors(guids)
    elif guids:
        unreal.WorldPartitionBlueprintLibrary.unload_actors(guids)
    after = {actor.get_path_name() for actor in level_actors()}
    changed = [str(d.actor_path) for d in targets if (str(d.actor_path) in after) == load]
    state = world_partition_state(after)
    return {"action": args["action"], "matched": len(descriptors), "requested": len(targets),
            "changed": len(changed), "not_changed": len(targets) - len(changed),
            "actors": changed[:args["listed"]], "loaded_actors": len(after),
            "actor_descriptors": state["actor_descriptors"], "unloaded_actors": state["unloaded_actors"]}

print(json.dumps(change()))
"""

# Expects ``args`` = {"actor_path", "width", "height", "format", "quality",
# "inline_limit"}. Renders what a camera or sensor actor sees into a
# width x height image: a transient SceneCapture2D stands where the actor's
# SceneCaptureComponent2D (preferred) or CameraComponent is, with its
# projection, field of view across the width (a CineCamera's from its focal
# length and filmback, get_horizontal_field_of_view: its field_of_view
# property can be stale, observed 90 for a 35 mm lens on UE 5.7.4), ortho
# width and post-process settings, hides the actor itself, captures final
# color (tone mapped, opaque alpha) once into a render target, and writes it
# with ImageWriteBlueprintLibrary.export_to_disk (PNG, or JPEG at ``quality``)
# into the editor's screenshot directory. Outside PIE the capture is spawned in
# the editor world through EditorActorSubsystem as a transient actor, which no
# listing shows and which leaves the map clean, and destroyed afterwards. While
# PIE runs the actor's counterpart in the game world (same path under the game
# world's package) is captured instead, through a capture spawned there
# (GameplayStatics' deferred spawn, which Unreal's Python reaches only through
# call_method) and destroyed; an actor without a counterpart there is refused,
# as the editor spawns nothing during PIE. The image is returned as base64 when
# it is at most ``inline_limit`` bytes (0: never).
CAMERA_CAPTURE = ACTOR_HELPERS + EDITOR_WORLD + """
import base64, math, os, time

def lens_of(actor):
    capture = actor.get_component_by_class(unreal.SceneCaptureComponent2D)
    if capture is not None:
        return {"component": capture, "kind": "SceneCaptureComponent2D",
                "orthographic": capture.projection_type == unreal.CameraProjectionMode.ORTHOGRAPHIC,
                "fov": capture.fov_angle, "ortho_width": capture.ortho_width,
                "post_process": capture.post_process_settings, "weight": capture.post_process_blend_weight,
                "aspect": None, "details": {}}
    camera = actor.get_component_by_class(unreal.CameraComponent)
    if camera is None:
        return None
    lens = {"component": camera, "kind": camera.get_class().get_name(),
            "orthographic": camera.projection_mode == unreal.CameraProjectionMode.ORTHOGRAPHIC,
            "fov": camera.field_of_view, "ortho_width": camera.ortho_width,
            "post_process": camera.post_process_settings, "weight": camera.post_process_blend_weight,
            "aspect": camera.aspect_ratio if camera.constrain_aspect_ratio else None, "details": {}}
    if isinstance(camera, unreal.CineCameraComponent):
        filmback = camera.filmback
        lens["fov"] = camera.get_horizontal_field_of_view()
        lens["details"] = {"focal_length_mm": camera.current_focal_length,
                           "sensor_width_mm": filmback.sensor_width, "sensor_height_mm": filmback.sensor_height}
    return lens

def counterpart(actor, editor, game):
    prefix = editor.get_path_name() if editor is not None else None
    path = actor.get_path_name()
    if prefix is None or not path.startswith(prefix + ":"):
        return None
    found = unreal.find_object(None, game.get_path_name() + path[len(prefix):])
    return found if isinstance(found, unreal.Actor) else None

def capture():
    source = actor_at(args["actor_path"])
    lens = lens_of(source)
    if lens is None:
        return {"success": False, "error_type": "ValidationError",
                "error": source.get_path_name() + " (a " + source.get_class().get_name() + ") has no camera or "
                         "scene capture component: pass a CameraActor, CineCameraActor, SceneCapture2D or an actor "
                         "with a CameraComponent or SceneCaptureComponent2D"}
    pie = pie_active()
    editor = editor_world()
    world, target, world_kind = editor, source, "editor"
    if pie:
        game = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
        target = counterpart(source, editor, game) if game is not None else None
        if target is None:
            return pie_refusal("capture an actor that is not in the running game")
        world, world_kind = game, "game"
        lens = lens_of(target)
    component = lens["component"]
    location, rotation = component.get_world_location(), component.get_world_rotation()
    transform = unreal.Transform(location=location, rotation=rotation, scale=unreal.Vector(1.0, 1.0, 1.0))
    if world_kind == "editor":
        spawned = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(
            unreal.SceneCapture2D, location, rotation, True)
        capture_component = spawned.capture_component2d if spawned is not None else None
    else:
        statics = unreal.get_default_object(unreal.GameplayStatics)
        spawned = statics.call_method("BeginDeferredActorSpawnFromClass",
                                      (world, unreal.SceneCapture2D.static_class(), transform))
        capture_component = spawned.capture_component2d if spawned is not None else None
    if capture_component is None:
        return {"success": False, "error_type": "EngineError", "error": "Unreal spawned no scene capture"}
    try:
        target_texture = unreal.RenderingLibrary.create_render_target2d(
            world, args["width"], args["height"], unreal.TextureRenderTargetFormat.RTF_RGBA8,
            unreal.LinearColor(0.0, 0.0, 0.0, 1.0), False)
        capture_component.set_editor_property("texture_target", target_texture)
        capture_component.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
        capture_component.set_editor_property("capture_every_frame", False)
        capture_component.set_editor_property("capture_on_movement", False)
        capture_component.set_editor_property("fov_angle", lens["fov"])
        capture_component.set_editor_property("ortho_width", lens["ortho_width"])
        capture_component.set_editor_property("projection_type", unreal.CameraProjectionMode.ORTHOGRAPHIC
                                              if lens["orthographic"] else unreal.CameraProjectionMode.PERSPECTIVE)
        capture_component.set_editor_property("post_process_settings", lens["post_process"])
        capture_component.set_editor_property("post_process_blend_weight", lens["weight"])
        capture_component.hide_actor_components(target, True)
        if world_kind == "game":
            spawned = statics.call_method("FinishSpawningActor", (spawned, transform))
        capture_component.capture_scene()
        directory = unreal.Paths.convert_relative_path_to_full(unreal.Paths.screen_shot_dir())
        os.makedirs(directory, exist_ok=True)
        label = re.sub("[^A-Za-z0-9_-]+", "_", target.get_actor_label())[:48] or "Camera"
        extension = "jpg" if args["format"] == "jpeg" else "png"
        path = os.path.join(directory, "SimulCamera_%s_%d.%s" % (label, int(time.time() * 1000), extension))
        options = unreal.ImageWriteOptions()
        options.format = unreal.DesiredImageFormat.JPG if extension == "jpg" else unreal.DesiredImageFormat.PNG
        options.compression_quality = args["quality"]
        options.async_ = False
        options.overwrite_file = True
        unreal.ImageWriteBlueprintLibrary.export_to_disk(target_texture, path, options)
    finally:
        if world_kind == "editor":
            unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(spawned)
        elif spawned is not None:
            spawned.destroy_actor()
    size = os.path.getsize(path) if os.path.isfile(path) else 0
    if not size:
        return {"success": False, "error_type": "CaptureError", "error": "Unreal wrote no image to " + path}
    result = {"path": path, "size_bytes": size, "actor_path": source.get_path_name(),
              "captured_actor": target.get_path_name(), "world": world_kind, "pie_active": pie,
              "component": lens["kind"], "orthographic": lens["orthographic"],
              "horizontal_fov": lens["fov"], "ortho_width": lens["ortho_width"],
              "constrained_aspect_ratio": lens["aspect"], "lens": lens["details"],
              "camera_location": [location.x, location.y, location.z],
              "camera_rotation": [rotation.pitch, rotation.yaw, rotation.roll]}
    if 0 < size <= args["inline_limit"]:
        with open(path, "rb") as stream:
            result["image_base64"] = base64.b64encode(stream.read()).decode("ascii")
    return result

print(json.dumps(capture()))
"""
