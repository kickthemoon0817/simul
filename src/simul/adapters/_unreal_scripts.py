"""Shared, fixed Python helpers executed inside the Unreal editor.

Caller data is embedded with repr/JSON, never interpolated as Python source.
These use the Python API so UObject references and world transforms do not
depend on Remote Control's property-description wire format.
"""

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

USD_IMPORT = """
import os, unreal, json
if not hasattr(unreal, "UsdStageImportFactory"):
    print(json.dumps({"success": False, "error_type": "PluginUnavailable",
        "error": "USD import requires the USDImporter plugin enabled at editor startup."}))
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

USD_EXPORT = """
import os, unreal, json
if not hasattr(unreal, "LevelExporterUSD"):
    print(json.dumps({"success": False, "error_type": "PluginUnavailable",
        "error": "USD export requires the USDImporter plugin enabled at editor startup."}))
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
ENGINE_METADATA = """
import json, unreal
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
print(json.dumps({
    "engine_version": unreal.SystemLibrary.get_engine_version(),
    "project_name": unreal.SystemLibrary.get_game_name(),
    "loaded_map": world.get_path_name() if world else "",
    "platform": unreal.GameplayStatics.get_platform_name(),
    # project_dir() is relative to the engine binaries; report the real path.
    "project_dir": unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()),
}))
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

# Expects ``args`` = {"path", "location", "rotation", "label"}: location
# [x, y, z] cm, rotation [pitch, yaw, roll] degrees, label None to keep the
# default. An Actor class (``/Script/Engine.PointLight``, a Blueprint's
# ``..._C`` class) spawns through spawn_actor_from_class. Any other asset goes
# through spawn_actor_from_object, the actor factory the editor uses when the
# asset is dropped into the level: a StaticMesh becomes a StaticMeshActor
# showing it, a Blueprint asset its actor. Spawn and label are one undoable
# transaction. When the path loads nothing, or Unreal places no actor for it,
# the script prints an error instead of an empty actor path.
SPAWN_ACTOR = ACTOR_HELPERS + """
def spawn(path, location, rotation, label):
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

# Expects ``args`` = {"label", "match_mode", "max_results"}. Semantic labels
# are actor tags. Matching ignores case, as Unreal's tag comparison (FName)
# does: "exact" compares whole tags, "contains" looks for the label inside a
# tag and "regex" searches each tag for the regular expression. Every match is
# counted; the first max_results are returned as actor_info entries.
FIND_ACTORS_BY_TAG = ACTOR_HELPERS + """
import re
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
for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    if any(matches(str(tag)) for tag in actor.get_editor_property("tags")):
        total += 1
        if len(actors) < args["max_results"]:
            actors.append(actor_info(actor))
print(json.dumps({"actors": actors, "total_matches": total,
                  "truncated": total > len(actors), "label_searched": label}))
"""

# Scene digest in one round trip. Class keys are UClass path names
# (``/Script/Engine.StaticMeshActor``), the same strings Remote Control's
# ``/remote/object/describe`` reports as ``Class``.
SCENE_SUMMARY = """
import json, unreal
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
counts = {}
total = 0
for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    name = actor.get_class().get_path_name()
    counts[name] = counts.get(name, 0) + 1
    total += 1
print(json.dumps({
    "map_path": world.get_path_name() if world else "",
    "total_actors": total,
    "actor_class_counts": counts,
}))
"""

# Expects ``args`` = {"parent", "name", "folder"}.
CREATE_MATERIAL_INSTANCE = """
import json, unreal
parent = unreal.EditorAssetLibrary.load_asset(args["parent"])
destination = args["folder"] + "/" + args["name"]
if parent is None or not isinstance(parent, unreal.MaterialInterface):
    print(json.dumps({"success": False, "error": "Parent material not found: " + args["parent"]}))
elif unreal.EditorAssetLibrary.does_asset_exist(destination):
    print(json.dumps({"success": False, "error": "Asset already exists: " + destination}))
else:
    # The factory's InitialParent is not exposed to Python; parent it after.
    instance = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        args["name"], args["folder"], unreal.MaterialInstanceConstant,
        unreal.MaterialInstanceConstantFactoryNew(),
    )
    if instance is None:
        print(json.dumps({"success": False, "error": "Could not create material instance: " + destination}))
    else:
        unreal.MaterialEditingLibrary.set_material_instance_parent(instance, parent)
        unreal.EditorAssetLibrary.save_loaded_asset(instance)
        print(json.dumps({
            "instance_path": instance.get_path_name(),
            "parent_path": parent.get_path_name(),
            "class_name": instance.get_class().get_name(),
        }))
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
