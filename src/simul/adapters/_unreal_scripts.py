"""Shared, fixed Python helpers executed inside the Unreal editor.

Caller data is embedded with repr/JSON, never interpolated as Python source.
These use the Python API so UObject references and world transforms do not
depend on Remote Control's property-description wire format.
"""

import hashlib
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

from ._unreal_errors import (
    ATTACHMENT_NOT_APPLIED,
    COMPONENT_NOT_APPLIED,
    DELETE_NOT_APPLIED,
    RENAME_NOT_APPLIED,
    VISIBILITY_NOT_APPLIED,
)

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
# be determined, and editor_world() returns None. get_editor_world() is asked
# only while no play world runs: during PIE it also logs "LogUtils: Error: The
# Editor is currently in a play mode." on every call (84 such lines in one
# session of tool calls, UE 5.7.4), which buried the editor's real errors.
EDITOR_WORLD_FUNCTION = """
def editor_world():
    editor = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    if editor.get_game_world() is None:
        return editor.get_editor_world()
    world = unreal.get_editor_subsystem(unreal.LayersSubsystem).get_world()
    if world is None or re.match("UEDPIE_[0-9]+_", world.get_outermost().get_name().rpartition("/")[2]):
        return None
    return world
"""

# PIE-aware helpers for the scripts below: pie_active(), pie_refusal(action)
# (the same envelope as the Python function), editor_world() and
# level_actors(). During PIE, level_actors() reads the editor world directly:
# every editable actor of its visible levels except the WorldSettings, which
# can include the builder brush and helper actors the Outliner hides. Either
# way each actor is listed once: after a USD import that replaced actors the
# level's actor list held each new actor twice (UE 5.7.4), and the robotics
# analysis counted 4 actors for 3.
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

def once_each(actors):
    seen, kept = set(), []
    for actor in actors:
        key = actor.get_path_name()
        if key not in seen:
            seen.add(key)
            kept.append(actor)
    return kept

def level_actors():
    if not pie_active():
        return once_each(unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors())
    world = editor_world()
    if world is None:
        return []
    return once_each(actor for actor in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
                     if actor.is_editable() and not isinstance(actor, unreal.WorldSettings))
"""
)

#: ``error_type`` of a call that acts on or reads the running simulation while
#: none runs.
SIMULATION_NOT_RUNNING = "SimulationNotRunning"

#: ``error_type`` of an actor with no copy in the running simulation.
NOT_IN_SIMULATION = "NotInSimulation"

#: The hint of a SimulationNotRunning refusal.
SIMULATION_START_HINT = (
    "Start the simulation with control_unreal_simulation, action 'start' (CLI: simul unreal sim start), "
    "then retry."
)

#: Why an editor actor can have no copy in the running simulation, for the
#: NotInSimulation refusal and the physics state read's ``warning``. In a World
#: Partition level Simulate streams in only the cells around the level
#: viewport's camera, so a body 1 km from it had no copy until
#: focus_unreal_on_actor moved the camera to it (UE 5.7.4).
NO_COPY_REASONS = (
    "an actor spawned after the simulation started, destroyed in it, or not loaded in it has none; in a World "
    "Partition level the simulation streams in only the cells around the level viewport's camera, so "
    "focus_unreal_on_actor on the actor (which moves the camera there) streams its cell in"
)

# The running simulation and an actor's copy in it, on top of EDITOR_WORLD:
# simulation_world(), in_simulation(actor), simulation_counterpart(actor) and
# editor_counterpart(actor). While Play In Editor (or Simulate In Editor) runs,
# Unreal duplicates the editor world into a play world whose package is the
# map's with UEDPIE_<n>_ before its name (/Game/Maps/UEDPIE_0_Lab), and each
# actor into it at the same path in that package, keeping its name, label and
# ActorGuid (observed on UE 5.7.4). The editor world's bodies read as
# simulating but never step: an impulse on the editor actor returned success
# while its copy stayed put, and reads gave the editor's z = 300 for a body
# resting at 49.5. So whatever acts on or reads the running simulation maps the
# actor to its copy first. An actor at a play-world path is its own copy, so
# simulation paths are accepted as they always were. When the path does not
# map (World Partition streams actors into cell levels), the copy is the play
# world's actor of the same class and ActorGuid. Lookups use find_object,
# which loads nothing. A copy destroyed during the simulation stays findable,
# invalid, until garbage collection (about a minute), so every counterpart is
# checked with SystemLibrary.is_valid: a destroyed one is no counterpart.
SIMULATION_WORLD = (
    EDITOR_WORLD
    + f"_SIMULATION_NOT_RUNNING, _NOT_IN_SIMULATION, _SIMULATION_START_HINT = "
    f"{SIMULATION_NOT_RUNNING!r}, {NOT_IN_SIMULATION!r}, {SIMULATION_START_HINT!r}\n"
    f"_NO_COPY_REASONS = {NO_COPY_REASONS!r}\n"
    + """
def simulation_world():
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()

def play_prefix(package_name):
    match = re.match("UEDPIE_[0-9]+_", package_name)
    return match.group(0) if match else None

def in_simulation(actor):
    world = actor.get_world()
    return world is not None and play_prefix(world.get_outermost().get_name().rpartition("/")[2]) is not None

def guid_of(actor):
    guid = actor.get_editor_property("actor_guid")
    return guid if unreal.GuidLibrary.is_valid_guid(guid) else None

def same_actor(candidate, actor, guid):
    return (candidate.get_class() == actor.get_class() and guid is not None
            and unreal.GuidLibrary.equal_equal_guid_guid(candidate.get_editor_property("actor_guid"), guid))

def live(actor):
    return isinstance(actor, unreal.Actor) and unreal.SystemLibrary.is_valid(actor)

def at_package(actor, rename):
    package, dot, rest = actor.get_path_name().partition(".")
    folder, slash, name = package.rpartition("/")
    renamed = rename(name)
    return None if renamed is None else unreal.find_object(None, folder + slash + renamed + dot + rest)

def simulation_counterpart(actor):
    world = simulation_world()
    if world is None or not live(actor):
        return None
    if in_simulation(actor):
        return actor
    prefix = play_prefix(world.get_outermost().get_name().rpartition("/")[2]) or ""
    copy = at_package(actor, lambda name: prefix + name)
    if live(copy) and copy.get_class() == actor.get_class():
        return copy
    guid = guid_of(actor)
    for candidate in unreal.GameplayStatics.get_all_actors_of_class(world, actor.get_class()):
        if live(candidate) and same_actor(candidate, actor, guid):
            return candidate
    return None

def editor_counterpart(actor):
    if not live(actor):
        return None
    if not in_simulation(actor):
        return actor
    original = at_package(actor, lambda name: name[len(play_prefix(name)):] if play_prefix(name) else None)
    if live(original) and not in_simulation(original):
        return original
    world = editor_world()
    guid = guid_of(actor)
    if world is None or guid is None:
        return None
    for candidate in unreal.GameplayStatics.get_all_actors_of_class(world, actor.get_class()):
        if live(candidate) and same_actor(candidate, actor, guid):
            return candidate
    return None

def simulation_refusal(what, why=None):
    error = "No simulation is running, and " + what + " needs the running simulation"
    if why:
        error += ": " + why
    return {"success": False, "error_type": _SIMULATION_NOT_RUNNING, "error": error,
            "details": {"code": "SIMULATION_NOT_RUNNING", "hint": _SIMULATION_START_HINT}}
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
# belongs to another world; for the PIE world's copy of an actor the message
# names the editor's path. Each raises SimulNotFound, whose message starts
# with ACTOR_NOT_FOUND, and the session reports it as NotFoundError.
# actor_at(path, play_world=True) also takes an actor of the running Play In
# Editor world (UnrealEditorSubsystem.get_game_world), for the physics and
# collision edits that change the running copy on purpose.
ACTOR_HELPERS = EDITOR_WORLD + f"""_ACTOR_NOT_FOUND = {ACTOR_NOT_FOUND!r}
_ATTACHMENT_NOT_APPLIED, _COMPONENT_NOT_APPLIED = {ATTACHMENT_NOT_APPLIED!r}, {COMPONENT_NOT_APPLIED!r}
_RENAME_NOT_APPLIED, _DELETE_NOT_APPLIED = {RENAME_NOT_APPLIED!r}, {DELETE_NOT_APPLIED!r}
_VISIBILITY_NOT_APPLIED = {VISIBILITY_NOT_APPLIED!r}
""" + """
def not_applied(error_type, message, expected, actual):
    return {"success": False, "error_type": error_type, "error": message,
            "details": {"expected": expected, "actual": actual}}
# The package name prefix of a Play In Editor world, UEDPIE_<n>_<map>.
_PIE_PACKAGE = "UEDPIE_[0-9]+_"

class SimulNotFound(Exception):
    pass

def actor_at(path, play_world=False):
    actor = unreal.find_object(None, path)
    world = editor_world()
    if not isinstance(actor, unreal.Actor) or not unreal.SystemLibrary.is_valid(actor):
        where = world.get_path_name() if world is not None else "no editor world"
        hint = ("; pass the actor's full object path, as list_unreal_actors reports it, not its label"
                if "/" not in path else "")
        raise SimulNotFound(_ACTOR_NOT_FOUND.format(path=path)
                            + " (no live actor at that path in the level open in the editor, " + where + hint + ")")
    owner = actor.get_world()
    if owner is not None and owner == world:
        return actor
    if (play_world and owner is not None
            and owner == unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()):
        return actor
    playing = owner is not None and re.match(_PIE_PACKAGE, owner.get_outermost().get_name().rpartition("/")[2])
    advice = ("Play In Editor's copy of an actor; pass the editor's, " + re.sub(_PIE_PACKAGE, "", path, count=1)
              if playing else "open that map to edit it")
    raise SimulNotFound(_ACTOR_NOT_FOUND.format(path=path) + " (it belongs to "
                        + (owner.get_path_name() if owner is not None else "no world")
                        + ", not to the level open in the editor, "
                        + (world.get_path_name() if world is not None else "none")
                        + "; " + advice + ")")

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

def editor_only(component):
    try:
        return bool(component.get_editor_property("is_editor_only"))
    except Exception:
        return False

def outside_root(actor):
    # The actor's primitive components (editor-only ones left out) not attached under its root
    # component, by name. Moving the actor moves its root's tree only, and Unreal's USD exporter
    # writes that tree only, so such a part stayed where it was and was left out of the stage
    # (UE 5.7.4): a part that set_simulate_physics detached in the editor, as simul's physics
    # tools did before they wrote the flag the way the Details panel does.
    root = actor.root_component
    left = []
    for component in actor.get_components_by_class(unreal.PrimitiveComponent):
        if component == root or editor_only(component):
            continue
        node = component.get_attach_parent()
        while node is not None and node != root:
            node = node.get_attach_parent()
        if node is None:
            left.append(component.get_name())
    return left

def outside_root_repair(names):
    return ("Reattach " + ("it" if len(names) == 1 else "them") + " under the actor's root: Undo the edit that "
            "detached " + ("it" if len(names) == 1 else "them") + " if it is still in the undo history, else "
            "execute_unreal_script with component.attach_to_component(actor.root_component, '', "
            "unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD, "
            "False) for each part attached to nothing.")
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

# Warnings and errors the editor logged while a script ran. Unreal's importers
# report what they skipped or could not read only to the editor log (and the
# message log, which Python cannot read), so the scripts that import read the
# lines the log file gained meanwhile, after FLUSHLOG writes out the buffered
# ones. They are whatever the editor logged in that time, so a line from other
# editor activity can appear. Defines log_mark() and logged_since(mark).
#
# The file is found by a marker this editor logs, not by name: a second editor
# of the same project writes <Project>_2.log beside <Project>.log, and the log
# named after the project alone was the other editor's, so its lines were
# reported for this one. log_mark() logs a unique line, flushes, and takes the
# project's log file (<Project>.log or <Project>_<n>.log) whose end holds it.
EDITOR_LOG = r"""
import os as _simul_os, re as _simul_re, uuid as _simul_uuid
_LOG_LINE = _simul_re.compile(r"^\[[^\]]*\]\[[^\]]*\](\w+): (Warning|Error): (.*)$")
_LOG_LIMIT = 20
_LOG_TAIL = 1 << 16

def _log_files():
    folder = unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_log_dir())
    name = _simul_re.compile(_simul_re.escape(unreal.SystemLibrary.get_game_name()) + r"(_[0-9]+)?\.log$")
    try:
        return [_simul_os.path.join(folder, entry) for entry in _simul_os.listdir(folder) if name.match(entry)]
    except OSError:
        return []

def log_mark():
    marker = "simul-log-mark-" + _simul_uuid.uuid4().hex
    unreal.log(marker)
    unreal.SystemLibrary.execute_console_command(None, "FLUSHLOG")
    token = marker.encode()
    for path in _log_files():
        try:
            size = _simul_os.path.getsize(path)
            with open(path, "rb") as handle:
                handle.seek(max(0, size - _LOG_TAIL))
                tail = handle.read()
        except OSError:
            continue
        at = tail.find(token)
        if at >= 0:
            return path, size - len(tail) + at + len(token)
    return None, None

def logged_since(mark):
    path, start = mark
    if path is None:
        return []
    unreal.SystemLibrary.execute_console_command(None, "FLUSHLOG")
    try:
        with open(path, "rb") as handle:
            handle.seek(start)
            text = handle.read(1 << 20).decode("utf-8", "replace")
    except OSError:
        return []
    found = []
    for line in text.splitlines():
        match = _LOG_LINE.match(line.strip())
        if match and match.group(1) != "LogPython":
            found.append(match.group(1) + ": " + match.group(2) + ": " + match.group(3).strip())
    return list(dict.fromkeys(found))[:_LOG_LIMIT]
"""

# Packages outside /Game the editor holds modified (dirty) in memory, which a
# Save All would write into the engine install or a plugin. Imports leave them
# so: on UE 5.7.4 Interchange set bUsedWithNanite on its shared parent material
# (/InterchangeAssets/Materials/PBRSurfaceMaterial_MR, "needed to have new
# flag set bUsedWithNanite"), leaving the plugin's package dirty. Unreal does
# that on a later editor frame, up to several imports after the one whose
# mesh needed it (observed: two frames after a reimport, seven after the first
# import), so no import can tell which one it was; each reports every such
# package the editor holds when it finishes. Python can neither save them
# selectively nor clear their dirty flag, so they are reported. Defines
# dirty_engine_packages() and dirty_engine_warning(names).
DIRTY_PACKAGES = """
def dirty_engine_packages():
    utils = unreal.EditorLoadingAndSavingUtils
    names = {package.get_name() for package in list(utils.get_dirty_map_packages())
             + list(utils.get_dirty_content_packages())}
    return sorted(name for name in names
                  if not (name == "/Game" or name.startswith("/Game/") or name.startswith("/Temp/")))

def dirty_engine_warning(names):
    return ("The editor holds " + str(len(names)) + " package(s) outside /Game modified in memory: "
            + ", ".join(names) + ". They belong to the engine or a plugin, shared by every project using it, "
            "and Unreal's importers mark them so, sometimes a moment after an import returns. simul saved none "
            "of them, and save_unreal_assets skips them, but the editor's Save All would write them into the "
            "engine install. Discard them when the editor asks, or restart it.")
"""

# Expects ``args`` = {"names": [...]}: relative file names an FBX stores for
# its textures. Prints where the editor opens each as given: from its working
# directory, as the FBX SDK's fopen does, and from its base directory, as
# Unreal's FPaths::ConvertRelativePathToFull does; import_asset holds both to
# the sandbox.
EDITOR_RELATIVE_PATHS = """
import json, os, unreal
print(json.dumps({"paths": {name: sorted({os.path.abspath(name), unreal.Paths.convert_relative_path_to_full(name)})
                            for name in args["names"]}}))
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
# the importer options (``properties`` by UsdStageImportOptions property
# name, ``fallback_collision_type`` as a UsdCollisionType name,
# ``prims_to_import`` and ``stage`` {meters_per_unit, up_axis}, each only when
# given) and whether to replace what an earlier import left. Imported actors
# are found as the level actors that were not there before, a listing Unreal
# withholds while PIE runs, so the import is refused then.
#
# The options are set on a fresh UsdStageImportOptions without change
# notifications. UsdStageImportOptions saves its config on every notified
# change (PostEditChangeProperty calls SaveConfig, UE 5.7), and
# set_editor_property notifies by default, so one import with import_actors
# false wrote bImportActors=False to the project's
# EditorPerProjectUserSettings.ini: every later import, the editor's own
# "Import into Level" dialog included, then spawned no actors. A fresh object
# starts from those saved settings, so every option simul documents is set on
# every import, to Unreal's default unless given. A saved setting that differs
# from Unreal's default is reported in warnings and left as it is: nothing
# tells a setting simul once wrote from one the user chose.
#
# Without replace the import is refused when the folder already holds assets,
# and runs with Unreal's Append actor policy (new actors beside existing ones)
# and Ignore asset policy (an existing asset is kept), so nothing is replaced.
# Unreal's defaults, Replace for both, replaced an earlier import's assets
# (a customized mesh lost its collision and lightmap resolution) and deleted
# its actors even though AssetImportTask.replace_existing was False, which the
# USD importer does not read. With replace, both policies are Replace.
#
# fallback_collision_type is the collision the importer gives a mesh whose
# prim has no UsdPhysics collision, and its default, none, gives none: an
# exported ground slab came back with no simple shape and a NoCollision
# component, so bodies would fall through it (UE 5.7.4). The new actors are
# checked (without_collision) and the reply lists those simulated bodies pass
# through, with a warning naming the options that give collision.
USD_IMPORT = EDITOR_WORLD + EDITOR_LOG + DIRTY_PACKAGES + """
import os

# UsdStageImportOptions' constructor defaults (UE 5.7) for the options simul sets on every import,
# enums by their lower-case name.
DEFAULTS = {
    "import_actors": True, "import_geometry": True, "import_materials": True,
    "import_skeletal_animations": True, "import_level_sequences": True, "import_groom_assets": True,
    "import_sparse_volume_textures": True, "import_sounds": True, "import_only_used_materials": False,
    "use_prim_kinds_for_collapsing": True, "kinds_to_collapse": 18, "merge_identical_material_slots": True,
    "share_assets_for_identical_prims": True, "prim_path_folder_structure": False, "interpret_lods": True,
    "nanite_triangle_threshold": 2147483647, "fallback_collision_type": "none", "override_stage_options": False,
}
CONFIG_NAMES = {"import_actors": "bImportActors", "import_geometry": "bImportGeometry",
                "import_materials": "bImportMaterials"}
LISTED = 20

def plain(value):
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(getattr(value, "name", value)).lower()

def import_options(values, replace):
    options = unreal.UsdStageImportOptions()
    quiet = unreal.PropertyAccessChangeNotifyMode.NEVER
    properties = {key: value for key, value in DEFAULTS.items()
                  if key not in ("fallback_collision_type", "override_stage_options")}
    properties.update(values.get("properties", {}))
    for key, value in properties.items():
        options.set_editor_property(key, value, quiet)
    collision = values.get("fallback_collision_type", DEFAULTS["fallback_collision_type"])
    options.set_editor_property("fallback_collision_type", getattr(unreal.UsdCollisionType, collision.upper()), quiet)
    if "prims_to_import" in values:
        options.set_editor_property("prims_to_import", values["prims_to_import"], quiet)
    stage = values.get("stage")
    options.set_editor_property("override_stage_options", stage is not None, quiet)
    if stage is not None:
        stage_options = unreal.UsdStageOptions()
        stage_options.set_editor_property("meters_per_unit", stage["meters_per_unit"])
        stage_options.set_editor_property("up_axis", getattr(unreal.UsdUpAxis, stage["up_axis"]))
        options.set_editor_property("stage_options", stage_options, quiet)
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
    return {key: plain(saved.get_editor_property(key)) for key in DEFAULTS}

def without_collision(actor_paths):
    # (mesh components, the actors with a mesh component simulated bodies pass through): one whose
    # collision has no physics, or whose mesh has no simple shape and does not stand its triangles in.
    tools = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    components, bare = 0, []
    for path in actor_paths:
        actor = unreal.find_object(None, path)
        if not isinstance(actor, unreal.Actor):
            continue
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh = component.get_editor_property("static_mesh")
            if mesh is None:
                continue
            components += 1
            physics = component.get_collision_enabled().name in ("QUERY_AND_PHYSICS", "PHYSICS_ONLY")
            shapes = tools.get_simple_collision_count(mesh) + tools.get_convex_collision_count(mesh)
            if not physics or (shapes == 0 and tools.get_collision_complexity(mesh).name
                               != "CTF_USE_COMPLEX_AS_SIMPLE"):
                bare.append(path)
                break
    return components, bare

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
    # A level's actor list can hold an actor twice after the importer has moved
    # actors into it (observed on UE 5.7.4), so each listing names each actor once.
    before = list(dict.fromkeys(actor.get_path_name() for actor in subsystem.get_all_level_actors()))
    mark = log_mark()
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
    logged = logged_since(mark)
    after = list(dict.fromkeys(actor.get_path_name() for actor in subsystem.get_all_level_actors()))
    known = set(before)
    actors = [p for p in after if p not in known]
    current = set(after)
    removed = [p for p in before if p not in current]
    if not assets and not actors:
        return {"success": False, "error_type": "EngineError",
                "error": "The USD importer produced no assets or actors from " + path
                         + (": " + "; ".join(logged) if logged else "; see the editor log"),
                "details": {"log": logged}}
    warnings = list(logged)
    dirtied = dirty_engine_packages()
    if dirtied:
        warnings.append(dirty_engine_warning(dirtied))
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
    meshes, bare = without_collision(actors)
    if bare:
        fallback = options.get("fallback_collision_type", DEFAULTS["fallback_collision_type"])
        warnings.append(
            str(len(bare)) + " of the imported actors have a mesh without physics collision (fallback_collision_type "
            + fallback + ", the collision the importer gives a mesh whose prim has no UsdPhysics collision), so "
            "simulated bodies fall through them and analyze_unreal_scene_for_robotics finds no floor among "
            "them; they are listed in actors_without_collision. Import again with replace_existing true and "
            "fallback_collision_type convex_decomposition (concave meshes, such as a floor with walls) or convex_hull "
            "(one hull per mesh), or give a mesh simple collision with compute_unreal_convex_hull (output collision).")
    return {"imported_assets": assets, "actor_paths": actors, "removed_actor_paths": removed,
            "warnings": warnings, "dirty_engine_packages": dirtied, "mesh_components": meshes,
            "actors_without_collision": bare[:LISTED], "actors_without_collision_count": len(bare)}

print(json.dumps(import_usd(**args)))
"""

# Expects ``args`` = {"path", "actors", "options", "overwrite"}. The export
# selects the actors and exports the editor world, neither of which Unreal
# allows while PIE runs, so the export is refused then. Each actor path must
# name an actor of the level being edited (actor_at). Unreal's USD level exporter writes
# what it accepts and leaves out the rest without an error (actors of another
# world, editor helpers such as a WorldSettings or PlayerStart), so the actors
# exported are read back from the written stage: the prim each actor's root
# component went to (UsdConversionLibrary.get_prim_path_for_object, which the
# exporter names it by, with the exporter's _<n> suffix for a name already
# used). An export that wrote none of them fails.
#
# The exporter writes a file per mesh and material into its asset folder,
# which is <output name>_assets beside the output, so that two exports into
# one folder do not write over each other's meshes; with export_sublayers it
# also writes a layer per sublevel and MaterialOverrides beside the output.
# Without ``overwrite`` the export is refused when any of those exists: the
# output, a file in its asset folder, or (with export_sublayers) a sublevel
# layer or MaterialOverrides. The files under the output's folder that are new
# or changed afterwards are reported as written; one that was there before is
# listed as overwritten. Unreal's exporter writes a DynamicMeshActor as an
# empty Xform, without its mesh, which is reported in warnings.
USD_EXPORT = ACTOR_HELPERS + """
import os

LISTED_FILES = 20

def files_under(folder):
    found = {}
    for root, _, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            found[path] = (stat.st_mtime_ns, stat.st_size)
    return found

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

def export_usd(path, actor_paths, values, overwrite):
    if not hasattr(unreal, "LevelExporterUSD"):
        return {"success": False, "error_type": "PluginUnavailable",
                "error": "USD export requires the USDImporter plugin enabled at editor startup."}
    if pie_active():
        return pie_refusal("export actors to USD")
    actors = [actor_at(actor_path) for actor_path in actor_paths]
    folder = os.path.dirname(path)
    stem, extension = os.path.splitext(os.path.basename(path))
    asset_folder = os.path.join(folder, stem + "_assets")
    beside = []
    if values.get("export_sublayers"):
        from pxr import Tf
        world = editor_world()
        beside.append(os.path.join(folder, "MaterialOverrides" + extension))
        levels = {actor.get_level().get_path_name(): actor.get_level() for actor in actors}
        for level in levels.values():
            outer = level.get_outer()
            if outer != world:
                beside.append(os.path.join(folder, Tf.MakeValidIdentifier(outer.get_name()) + extension))
    if not overwrite:
        existing = [p for p in [path] + beside if os.path.exists(p)] + sorted(files_under(asset_folder))
        if existing:
            return {"success": False, "error_type": "InvalidState",
                    "error": ("The export would write over " + str(len(existing)) + " existing file(s) and "
                              "overwrite is false, so nothing was exported: " + ", ".join(existing[:5])),
                    "details": {"existing_files": existing[:LISTED_FILES],
                                "hint": "Pass another output_path, or overwrite=true to replace them"}}
    before = files_under(folder)
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
    inner.asset_folder = unreal.DirectoryPath(asset_folder)
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
        os.makedirs(asset_folder, exist_ok=True)
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
        try:
            os.rmdir(asset_folder)  # only when the export put nothing in it
        except OSError:
            pass
    after = files_under(folder)
    written = sorted(p for p, stamp in after.items() if before.get(p) != stamp)
    size = os.path.getsize(path) if os.path.isfile(path) else 0
    if size == 0:
        return {"success": False, "error_type": "EngineError", "error": "USD export wrote no file at " + path,
                "details": {"written_files": written}}
    exported, skipped = exported_prims(path, actors, inner)
    result = {"output_path": path, "actors_exported": len(exported), "exported_prims": exported,
              "skipped_actors": skipped, "file_size_bytes": size, "written_files": written,
              "overwritten_files": [p for p in written if p in before], "warnings": []}
    meshless = [actor.get_path_name() for actor in actors
                if hasattr(unreal, "DynamicMeshComponent")
                and actor.get_components_by_class(unreal.DynamicMeshComponent)]
    if meshless:
        result["warnings"].append("Unreal's USD exporter writes DynamicMeshActors as empty Xforms, without their "
                                  "mesh: " + ", ".join(meshless) + ". convert_unreal_mesh_format with target_format "
                                  "static_mesh and replace_actor true makes them StaticMeshActors, which export "
                                  "with their geometry.")
    left_out = [{"actor_path": actor.get_path_name(), "components": outside_root(actor)} for actor in actors]
    result["components_left_out"] = [entry for entry in left_out if entry["components"]][:LISTED_FILES]
    if result["components_left_out"]:
        names = [name for entry in result["components_left_out"] for name in entry["components"]]
        result["warnings"].append(
            "Unreal's USD exporter writes an actor's components from its root component down, so it left out the "
            "parts not attached under their actor's root (components_left_out): "
            + "; ".join(entry["actor_path"] + ": " + ", ".join(entry["components"])
                        for entry in result["components_left_out"]) + ". " + outside_root_repair(names)
            + " Then export again.")
    if not exported:
        result.update(success=False, error_type="EngineError",
                      error="Unreal's USD exporter wrote no prim for any actor given (it leaves out editor "
                            "helpers such as WorldSettings, PlayerStart and brushes): " + ", ".join(skipped))
    return result

print(json.dumps(export_usd(args["path"], args["actors"], args["options"], args["overwrite"])))
"""

# Interchange's formats for each kind of translator (Unreal 5.7 has ASSETS,
# SCENES and ACTORS) and the pipeline stacks the project's Interchange settings
# define for content and scene imports, with each stack's pipeline assets.
INTERCHANGE_INFO = """
import json, unreal

def stacks(settings, kind):
    return [{"name": str(name), "kind": kind,
             "pipelines": [pipeline.export_text() for pipeline in stack.get_editor_property("pipelines")]}
            for name, stack in settings.get_editor_property("pipeline_stacks").items()]

def info():
    if not hasattr(unreal, "InterchangeManager"):
        return {"success": False, "error_type": "PluginUnavailable",
                "error": "Interchange is not available in this editor"}
    manager = unreal.InterchangeManager.get_interchange_manager_scripted()
    by_kind = {}
    for kind in ("ASSETS", "SCENES", "ACTORS"):
        translator = getattr(unreal.InterchangeTranslatorType, kind, None)
        if translator is not None:
            by_kind[kind.lower()] = sorted({str(f) for f in manager.get_supported_formats(translator)})
    project = unreal.get_default_object(unreal.InterchangeProjectSettings)
    content = project.get_editor_property("content_import_settings")
    scene = project.get_editor_property("scene_import_settings")
    return {
        "pipelines": stacks(content, "content") + stacks(scene, "scene"),
        "pipeline_enumeration_available": True,
        "default_pipeline_stacks": {"content": str(content.get_editor_property("default_pipeline_stack")),
                                    "scene": str(scene.get_editor_property("default_pipeline_stack"))},
        "supported_formats": sorted({f for formats in by_kind.values() for f in formats}),
        "supported_formats_by_kind": by_kind,
        "interchange_version": unreal.SystemLibrary.get_engine_version(),
        "usd_import_available": hasattr(unreal, "UsdStageImportFactory"),
        "usd_export_available": hasattr(unreal, "LevelExporterUSD"),
        "simready_available": False,
    }

print(json.dumps(info()))
"""

# Expects ``args`` = {"path", "destination", "replace", "sources"}. Imports an
# FBX, glTF, GLB or OBJ file as assets through Interchange, Unreal 5.7's
# importer for them, with the project's default content pipelines, the way the
# Content Browser's Import does: no actor is placed. The import is synchronous
# (AssetImportTask.async_ is false) and saves what it creates.
#
# What an earlier import of the same file left is reported as kept. An asset
# in the destination folder came from this file when its import data names
# one of ``sources``: the file itself (meshes, materials) or a file it
# references (textures), resolved on the simul host. Only an asset whose
# registry tag (AssetImportData, which loads nothing) mentions one of their
# file names is loaded to read its import data exactly. Interchange keeps an
# existing asset when replace_existing is false, and also when it is true for
# all but the asset named after the file: replace_existing turns the import
# into a reimport of that one asset (UInterchangeManager's convert-to-reimport),
# which leaves the textures and materials it made before as they were, even
# when their source changed (observed on UE 5.7.4: a 32 px texture redrawn at
# 48 px stayed 32 px). Interchange also logs "The option bReplaceExisting is
# false, we are not overriding the asset named '<class> <path>'", which is
# read back too. A file whose every asset already exists imports nothing,
# which is reported as InvalidState rather than as a failed import.
ASSET_IMPORT = EDITOR_WORLD + EDITOR_LOG + DIRTY_PACKAGES + """
import os, re

KEPT = re.compile(r"we are not overriding the asset named '(?:\\S+ )?([^']+)'")

def same_file(path):
    return os.path.normcase(os.path.realpath(path))

def import_sources(asset, names):
    tag = str(asset.get_tag_value("AssetImportData") or "").lower()
    if not any(name in tag for name in names):
        return []
    try:
        loaded = unreal.load_asset(str(asset.package_name) + "." + str(asset.asset_name))
        data = loaded.get_editor_property("asset_import_data") if loaded is not None else None
        return [same_file(name) for name in data.extract_filenames()] if data is not None else []
    except Exception:
        return []

def import_asset(path, destination, replace, sources):
    if pie_active():
        return pie_refusal("import assets")
    if not os.path.isfile(path):
        return {"success": False, "error_type": "NotFoundError", "error": "File not found on the editor host: " + path}
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    wanted = {same_file(source) for source in sources}
    names = {os.path.basename(source).lower() for source in sources}
    before, from_this_file = {}, []
    for asset in registry.get_assets_by_path(destination, recursive=True):
        package = str(asset.package_name)
        before[package] = package + "." + str(asset.asset_name)
        if wanted.intersection(import_sources(asset, names)):
            from_this_file.append(before[package])
    mark = log_mark()
    task = unreal.AssetImportTask()
    try:
        task.filename = path
        task.destination_path = destination
        task.automated = True
        task.save = True
        task.replace_existing = replace
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        imported = [str(p) for p in task.imported_object_paths]
    finally:
        task.factory = None
        task.options = None
    logged = logged_since(mark)
    dirtied = dirty_engine_packages()
    done = set(imported)
    kept = sorted({match.group(1) for line in logged for match in [KEPT.search(line)] if match}
                  | {found for found in from_this_file if found not in done})
    warnings = list(logged)
    if dirtied:
        warnings.append(dirty_engine_warning(dirtied))
    if not imported:
        if kept:
            return {"success": False, "error_type": "InvalidState",
                    "error": ("Every asset " + os.path.basename(path) + " would create already exists and "
                              "replace_existing is false, so nothing was imported: " + ", ".join(kept)),
                    "details": {"existing_assets": kept, "dirty_engine_packages": dirtied,
                                "hint": "Pass another destination_path, or replace_existing=true to replace them"}}
        return {"success": False, "error_type": "EngineError",
                "error": "Unreal imported nothing from " + path + (": " + "; ".join(logged) if logged else ""),
                "details": {"log": logged, "dirty_engine_packages": dirtied}}
    replaced = [p for p in imported if p.split(".", 1)[0] in before]
    if replace and kept:
        warnings.append(
            "replace_existing reimported " + ", ".join(replaced or imported) + ", but Unreal's Interchange "
            "reimport keeps the other assets an earlier import of this file made, as they were: "
            + ", ".join(kept) + ". Delete them, or import into another destination_path, to import them anew.")
    return {"imported_assets": imported, "replaced_assets": replaced, "kept_assets": kept,
            "warnings": warnings, "dirty_engine_packages": dirtied}

print(json.dumps(import_asset(**args)))
"""

# Expects ``args`` = {"asset_paths", "save_level", "all_dirty",
# "allow_engine_content", "dry_run"}. Saves modified packages: the level being
# edited (its map package and, in a World Partition level, the packages of its
# actors and objects), every modified package, and the packages of the assets
# named, each only when it is modified. A package outside /Game (engine or
# plugin content, shared by every project using it) is skipped unless
# allow_engine_content, as is one never saved, under /Temp, which has no file
# to save to. The editor saves nothing while PIE runs, so that is refused.
# What is still modified afterwards failed to save.
SAVE_ASSETS = EDITOR_WORLD + """
def dirty_packages():
    utils = unreal.EditorLoadingAndSavingUtils
    found = {}
    for package in list(utils.get_dirty_map_packages()) + list(utils.get_dirty_content_packages()):
        found[package.get_name()] = package
    return found

def of_level(name, level):
    if name == level:
        return True
    mount, _, rest = level[1:].partition("/")
    return any(name.startswith("/" + mount + "/" + folder + "/" + rest + "/")
               for folder in ("__ExternalActors__", "__ExternalObjects__"))

def save(asset_paths, save_level, all_dirty, allow_engine_content, dry_run):
    if pie_active():
        return pie_refusal("save assets")
    dirty = dirty_packages()
    world = editor_world()
    level = world.get_package().get_name() if world is not None else None
    chosen, skipped = {}, []
    if save_level:
        if level is None or level.startswith("/Temp/"):
            return {"success": False, "error_type": "InvalidState",
                    "error": "The level being edited has never been saved, so it has no file to save to",
                    "details": {"level_package": level,
                                "hint": "Save it once in the editor (File > Save Current Level As) first"}}
        for name in dirty:
            if of_level(name, level):
                chosen[name] = "level"
    if all_dirty:
        for name in dirty:
            chosen.setdefault(name, "modified")
    for path in asset_paths:
        name = path.split(".", 1)[0]
        if name in dirty:
            chosen.setdefault(name, "asset")
        else:
            skipped.append({"package": name, "reason": "not modified, or not loaded"})
    targets = {}
    for name, why in sorted(chosen.items()):
        if name.startswith("/Temp/"):
            skipped.append({"package": name, "reason": "never saved, so it has no file to save to"})
        elif not allow_engine_content and not (name == "/Game" or name.startswith("/Game/")):
            skipped.append({"package": name, "reason": "outside /Game (engine or plugin content); pass "
                                                         "allow_engine_content=true to save it"})
        else:
            targets[name] = why
    if dry_run or not targets:
        return {"saved": [], "failed": [], "would_save": sorted(targets), "skipped": skipped,
                "level_package": level, "dry_run": dry_run}
    unreal.EditorLoadingAndSavingUtils.save_packages([dirty[name] for name in targets], True)
    still = dirty_packages()
    failed = [name for name in targets if name in still]
    result = {"saved": [name for name in targets if name not in still], "failed": failed,
              "would_save": [], "skipped": skipped, "level_package": level, "dry_run": False}
    if failed:
        result.update(success=False, error_type="EngineError",
                      error="Unreal did not save " + ", ".join(failed) + "; see the editor log")
    return result

print(json.dumps(save(**args)))
"""

# Expects ``args`` = {"query", "class_names", "package_paths",
# "recursive_paths", "include_subclasses", "offset", "limit"}. Searches the
# asset registry, in or out of PIE, without loading any asset. Remote
# Control's asset search, which the tool used before, took only a class path
# (a short name such as StaticMesh matched nothing and fired an engine ensure,
# 15.7 s on the first call), searched a package path without its subfolders,
# searched only /Game without a filter, stopped at the limit (so whether more
# matched was unknown) and named no package path (observed on UE 5.7.4).
# Here a class is given by its short name (resolved through the editor's
# Python classes) or its path; package paths include their subfolders unless
# recursive_paths is false; without package paths every mounted content root
# is searched (/Game, /Engine and every plugin's); the query matches part of
# the asset name, ignoring case; redirectors are left out. The matches are
# sorted by object path, ignoring case, so offset pages through them, and
# ``total`` counts them all. Each asset carries the registry tags Unreal
# records for its kind (TAGS), as the strings Unreal stores.
ASSET_SEARCH = """
import json, unreal

TAGS = (
    "Triangles", "Vertices", "LODs", "Materials", "UVChannels", "NaniteEnabled", "NaniteTriangles",
    "CollisionPrims", "CollisionComplexity", "DefaultCollision", "SectionsWithCollision", "ApproxSize",
    "Bones", "MorphTargets", "Skeleton", "PhysicsAsset",
    "Dimensions", "Format", "SRGB", "CompressionSettings", "HasAlphaChannel",
    "MaterialDomain", "BlendMode", "ShadingModel", "Parent",
    "Duration", "NumChannels", "SampleRate", "SequenceLength", "Number of Frames",
    "ParentClass", "NativeParentClass", "GeneratedClass",
)

def class_path(name):
    if name.startswith("/"):
        package, dot, asset = name.rpartition(".")
        return package + "." + asset if dot and package and asset else None
    cls = getattr(unreal, name, None)
    static = getattr(cls, "static_class", None) if isinstance(cls, type) else None
    return static().get_path_name() if static is not None else None

def top_level(path):
    package, _, asset = path.rpartition(".")
    return unreal.TopLevelAssetPath(package, asset)

def entry(asset):
    package, name, kind = str(asset.package_name), str(asset.asset_name), asset.asset_class_path
    metadata = {}
    for tag in TAGS:
        value = asset.get_tag_value(tag)
        if value is not None:
            metadata[tag] = value
    return {"name": name, "path": package + "." + name, "class_name": str(kind.asset_name),
            "class_path": str(kind.package_name) + "." + str(kind.asset_name), "package_name": package,
            "package_path": str(asset.package_path), "metadata": metadata}

def search(query, class_names, package_paths, recursive_paths, include_subclasses, offset, limit):
    resolved, unknown = [], []
    for name in class_names:
        path = class_path(name)
        if path is None:
            unknown.append(name)
        else:
            resolved.append(path)
    if unknown:
        return {"success": False, "error_type": "ValidationError",
                "error": ("Unknown asset class: " + ", ".join(unknown) + ". Pass a class name such as "
                          "StaticMesh, Material or Texture2D, or a class path such as /Script/Engine.StaticMesh")}
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    if resolved or package_paths:
        found = registry.get_assets(unreal.ARFilter(
            class_paths=[top_level(path) for path in resolved], package_paths=package_paths,
            recursive_paths=recursive_paths, recursive_classes=include_subclasses))
    else:
        found = registry.get_all_assets()
    needle = query.lower()
    matches = {}
    for asset in found:
        name = str(asset.asset_name)
        if asset.is_redirector() or (needle and needle not in name.lower()):
            continue
        matches[str(asset.package_name) + "." + name] = asset
    ordered = sorted(matches, key=str.lower)
    page = ordered[offset:offset + limit]
    more = offset + len(page) < len(ordered)
    return {"assets": [entry(matches[path]) for path in page], "count": len(page), "total": len(ordered),
            "offset": offset, "next_offset": offset + len(page) if more else None, "truncated": more,
            "class_paths": resolved}

print(json.dumps(search(**args)))
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
left = outside_root(actor) if values else []
warning = None
if left:
    warning = (", ".join(left) + (" is" if len(left) == 1 else " are") + " not attached under the actor's root "
               "component, so " + ("it" if len(left) == 1 else "they") + " did not move with it. "
               + outside_root_repair(left))
print(json.dumps({"actor_path": actor.get_path_name(), **actor_transform(actor), "warning": warning}))
"""

# Expects ``args`` = {"actor_path"}. Makes the actor the whole editor
# selection (adding it left earlier focuses selected, UE 5.7.4), reads the
# selection back and prints the actor's location and bounds for the camera.
# The bounds are AActor::GetActorBounds over every primitive component, its
# child actors' included, as a center and a half-size. The half-size is
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
    subsystem.set_selected_level_actors([actor])
    selected = actor in subsystem.get_selected_level_actors()
origin, extent = focus_bounds(actor)
print(json.dumps({"location": actor_transform(actor)["location"],
                  "bounds_origin": origin, "bounds_extent": extent,
                  "selected": selected, "pie_active": pie}))
"""

#: Actor classes ``spawn_unreal_actor`` refuses, each with why, subclasses
#: included: the engine's per-level singletons, which it creates with the
#: level or the world, and the game-framework classes the game mode spawns when
#: play starts. All are declared NotPlaceable in UE 5.7, which the editor's
#: Place Actors panel honours and EditorActorSubsystem's spawns do not (a
#: second WorldSettings landed in a saved level, UE 5.7.4); Unreal's Python API
#: reports no class flags, so the list is fixed here.
NOT_PLACEABLE_ACTOR_CLASSES = (
    ("/Script/Engine.WorldSettings", "each level has exactly one WorldSettings, which Unreal creates with it"),
    ("/Script/Engine.LevelScriptActor", "the level Blueprint's actor, which Unreal creates with the level"),
    ("/Script/Engine.WorldDataLayers", "the level's data layer registry, which Unreal creates with the level"),
    ("/Script/Engine.WorldPartitionMiniMap", "the World Partition minimap, which Unreal creates with the level"),
    ("/Script/Engine.DefaultPhysicsVolume", "the world's default physics volume, which Unreal creates"),
    ("/Script/Engine.ParticleEventManager", "the world's particle event manager, which Unreal creates"),
    ("/Script/NavigationSystem.NavigationData", "navigation data, which Unreal builds from a NavMeshBoundsVolume"),
    ("/Script/Engine.GameModeBase", "a game-framework class, which the game mode spawns when play starts"),
    ("/Script/Engine.GameStateBase", "a game-framework class, which the game mode spawns when play starts"),
    ("/Script/Engine.GameSession", "a game-framework class, which the game mode spawns when play starts"),
    ("/Script/Engine.GameNetworkManager", "a game-framework class, which the game mode spawns when play starts"),
    ("/Script/Engine.PlayerState", "a game-framework class, which the game mode spawns when play starts"),
    ("/Script/Engine.Controller", "a game-framework class, which the game mode spawns when play starts"),
    ("/Script/Engine.HUD", "a game-framework class, which the game mode spawns when play starts"),
    ("/Script/Engine.PlayerCameraManager", "a game-framework class, which the game mode spawns when play starts"),
)

#: Why ``spawn_unreal_actor`` refuses the bare ``Brush`` class (its subclasses,
#: the volumes, spawn): a brush spawned without geometry is the builder
#: brush's class, and placing one rebuilt BSP for 14 s in a World Partition
#: level (UE 5.7.4).
BARE_BRUSH_REASON = (
    "a bare BSP brush, which has no geometry; Unreal makes brushes with its brush tools. Spawn a volume "
    "class such as /Script/Engine.BlockingVolume, or a mesh such as /Engine/BasicShapes/Cube.Cube"
)

# Expects ``args`` = {"path", "location", "rotation", "label"}: location
# [x, y, z] cm, rotation [pitch, yaw, roll] degrees, label None to keep the
# default. An Actor class (``/Script/Engine.PointLight``, a Blueprint's
# ``..._C`` class) spawns through spawn_actor_from_class. Any other asset goes
# through spawn_actor_from_object, the actor factory the editor uses when the
# asset is dropped into the level: a StaticMesh becomes a StaticMeshActor
# showing it, a Blueprint asset its actor. Before anything loads, a path into a
# package (an actor in a level) and a map are refused: loading either loads a
# map. A class Unreal does not let you place (NOT_PLACEABLE_ACTOR_CLASSES, the
# bare Brush), or a Blueprint of one, is refused before the spawn. Spawn and
# label are one undoable transaction, and the label is read back: one Unreal
# does not take (it trims spaces and refuses "None" and labels of 1024
# characters or more, without a word) removes the actor and cancels the
# transaction. When the path loads nothing, or Unreal places no actor for it,
# the script prints an error instead of an empty actor path; while PIE runs,
# when Unreal places nothing whatever the path, it prints the PIE refusal.
SPAWN_ACTOR = ACTOR_HELPERS + f"""
NOT_PLACEABLE = {NOT_PLACEABLE_ACTOR_CLASSES!r}
BARE_BRUSH = {BARE_BRUSH_REASON!r}
""" + """
def refused(error_type, message):
    return {"success": False, "error_type": error_type, "error": message}

def placement_refusal(source, path):
    cls = source.generated_class() if isinstance(source, unreal.Blueprint) else source
    if not isinstance(cls, unreal.Class):
        return None
    if cls == unreal.Brush.static_class():
        return path + " is " + BARE_BRUSH
    for class_path, reason in NOT_PLACEABLE:
        base = unreal.find_object(None, class_path)
        if base is not None and unreal.MathLibrary.class_is_child_of(cls, base):
            return (path + " is " + ("" if cls == base else "a subclass of ") + base.get_name() + ": " + reason
                    + "; Unreal does not let you place it")
    return None

def spawn(path, location, rotation, label):
    if pie_active():
        return pie_refusal("spawn actors")
    if ":" in path:
        return refused("SpawnError", path + " names an object inside a package, such as an actor in a level, "
                       "not a class or asset")
    if not path.startswith("/Script/"):
        assets = unreal.AssetRegistryHelpers.get_asset_registry().get_assets_by_package_name(path.split(".", 1)[0])
        if any(str(asset.asset_class_path.asset_name) == "World" for asset in assets):
            return refused("SpawnError", path + " is a map, not a class or asset; simul loads no map")
    source = unreal.load_object(None, path)
    if source is None:
        return {"success": False, "error_type": "NotFoundError",
                "error": "No class or asset at " + repr(path) + ": pass an Actor class path "
                         "such as /Script/Engine.PointLight or an asset path such as "
                         "/Engine/BasicShapes/Cube.Cube"}
    if isinstance(source, unreal.Actor):
        return refused("SpawnError", path + " is an actor in a level, not a class or asset")
    is_class = isinstance(source, unreal.Class)
    if is_class and not unreal.MathLibrary.class_is_child_of(source, unreal.Actor.static_class()):
        return refused("SpawnError", path + " is a class but not an Actor class")
    reason = placement_refusal(source, path)
    if reason is not None:
        return refused("SpawnError", reason)
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    location = unreal.Vector(*location)
    pitch, yaw, roll = rotation
    rotation = unreal.Rotator(pitch=pitch, yaw=yaw, roll=roll)
    with unreal.ScopedEditorTransaction("simul: spawn actor") as transaction:
        if is_class:
            actor = subsystem.spawn_actor_from_class(source, location, rotation)
        else:
            actor = subsystem.spawn_actor_from_object(source, location, rotation)
        if actor is not None and label is not None:
            actor.set_actor_label(label)
            got = actor.get_actor_label()
            if got != label:
                subsystem.destroy_actor(actor)
                transaction.cancel()
                return not_applied(_RENAME_NOT_APPLIED, "Unreal did not take the label " + repr(label[:80])
                                   + ", so the actor was removed again", {"label": label}, {"label": got})
    if actor is None:
        return refused("SpawnError", "Unreal placed no actor for " + path + " (a "
                       + source.get_class().get_name() + "); pass a concrete Actor class (Unreal "
                       "spawns no abstract class, such as Light or Info) or an asset the editor can "
                       "place, such as a StaticMesh or a Blueprint")
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
        return not_applied(_ATTACHMENT_NOT_APPLIED, "Unreal did not detach " + actor_path + " from " + path_of(after),
                           {"parent_path": None}, {"parent_path": path_of(after)})
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
        return not_applied(_ATTACHMENT_NOT_APPLIED, "Unreal did not attach " + actor_path + " to " + parent_path
                           + "; the editor log says why", {"parent_path": parent.get_path_name()},
                           {"parent_path": path_of(after)})
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
            return not_applied(_COMPONENT_NOT_APPLIED, "Unreal added no " + cls.get_name() + " to " + actor_path
                               + (": " + str(failure) if str(failure) else ""),
                               {"component_class": cls.get_name()}, {"component_path": None})
        if component_name is not None and not (subsystem.rename_subobject(handle, unreal.Text(component_name))
                                               and component.get_name() == component_name):
            got = component.get_name()
            subsystem.delete_subobject(handles[0], handle, None)
            transaction.cancel()
            return not_applied(_RENAME_NOT_APPLIED, "Unreal did not name the new " + cls.get_name() + " "
                               + repr(component_name) + ", so it was removed again",
                               {"component_name": component_name}, {"component_name": got})
    if component not in actor.get_components_by_class(unreal.ActorComponent):
        return not_applied(_COMPONENT_NOT_APPLIED, "The new " + cls.get_name() + " is not among " + actor_path
                           + "'s components", {"component_path": component.get_path_name()}, {"component_path": None})
    return dict(describe(actor, component), actor_path=actor_path)

print(json.dumps(add(**args)))
"""

# Expects ``args`` = {"actor_path"}. Raises SimulNotFound unless the path
# names a live actor of the editor world; prints an empty object otherwise.
# For an operation Remote Control performs on the object path itself, which
# would act on a deleted actor, or on an actor of a map loaded but not open.
ACTOR_EXISTS = ACTOR_HELPERS + """
actor_at(args["actor_path"])
print("{}")
"""

# Expects ``args`` = {"actor_path"}. Deletes the actor in one undoable
# transaction and reads its validity back. While PIE runs Unreal deletes
# nothing, so the PIE refusal comes first. A path that names no live actor of
# the editor world, a label among them, is NotFoundError: it used to be
# success with deleted false. Unreal's level listing leaves out the actors the
# editor does not let you delete (the WorldSettings, the builder brush, actors
# kept out of the Outliner), and EditorActorSubsystem.destroy_actor deletes
# them anyway, so one the listing leaves out is refused.
DELETE_ACTOR = ACTOR_HELPERS + """
def delete(actor_path):
    if pie_active():
        return pie_refusal("delete actors")
    try:
        actor = actor_at(actor_path)
    except SimulNotFound as exc:
        return {"success": False, "error_type": "NotFoundError", "error": str(exc)}
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if actor not in subsystem.get_all_level_actors():
        if isinstance(actor, unreal.WorldSettings):
            kind = "the level's WorldSettings"
        elif actor.get_class() == unreal.Brush.static_class():
            kind = "the level's builder brush"
        else:
            kind = "an actor the editor keeps out of the Outliner"
        return {"success": False, "error_type": "ValidationError",
                "error": actor_path + " is " + kind + ", which the editor does not let you delete"}
    with unreal.ScopedEditorTransaction("simul: delete actor") as transaction:
        destroyed = subsystem.destroy_actor(actor)
        if not destroyed:
            transaction.cancel()
    if not destroyed or unreal.SystemLibrary.is_valid(actor):
        return not_applied(_DELETE_NOT_APPLIED, "Unreal did not delete " + actor_path, {"deleted": True},
                           {"deleted": False})
    return {"actor_path": actor_path, "deleted": True}

print(json.dumps(delete(**args)))
"""

# Expects ``args`` = {"actor_path", "visible", "scope", "propagate",
# "persistent"}. Three flags hide an actor (UE 5.7.4), and the scope names the
# ones written:
# - "editor": the editor's hide (set_is_temporarily_hidden_in_editor, the
#   Outliner's eye and the H key), which the level viewport and its captures
#   honour. Unreal keeps it for the editor session, not in the saved level.
# - "game": hidden in game (set_actor_hidden_in_game, bHidden), saved with the
#   level, which Play In Editor and a packaged game honour; the level viewport
#   still draws the actor outside Game View.
# - "all": both.
# With persistent (editor scopes) the editor's hidden-at-startup flag
# (bHiddenEd, saved with the level and applied when it loads) is written too,
# through the editor's ACTOR HIDE / UNHIDE SELECTED STARTUP commands: Python
# and Remote Control cannot set it (the property is not exposed and is
# EditConst). They act on the selection, so the actors are selected for the
# command, made selectable first (the editor selects no hidden actor), and the
# selection is restored; the editor selects nothing during PIE, so persistent
# is refused then. A hidden parent hides none of its attached actors, so
# propagate applies the same flags to every actor attached under it, children
# of children included. Everything is one undoable transaction with every actor
# modified first, and every actor's flags are read back.
SET_ACTOR_VISIBILITY = ACTOR_HELPERS + """
STARTUP_COMMANDS = {True: "ACTOR HIDE SELECTED STARTUP", False: "ACTOR UNHIDE SELECTED STARTUP"}

def flags(actor):
    return {"actor_path": actor.get_path_name(),
            "hidden_in_editor": bool(actor.is_temporarily_hidden_in_editor()),
            "hidden_in_editor_at_startup": bool(actor.is_hidden_ed_at_startup()),
            "hidden_in_game": bool(actor.get_editor_property("hidden"))}

def set_hidden_at_startup(actors, hidden):
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    previous = list(subsystem.get_selected_level_actors())
    for actor in actors:
        if actor.is_temporarily_hidden_in_editor():
            actor.set_is_temporarily_hidden_in_editor(False)
    try:
        subsystem.set_selected_level_actors(actors)
        unreal.SystemLibrary.execute_console_command(None, STARTUP_COMMANDS[hidden])
    finally:
        subsystem.set_selected_level_actors(previous)

def set_visibility(actor_path, visible, scope, propagate, persistent):
    actor = actor_at(actor_path)
    if persistent and pie_active():
        return pie_refusal("change what is hidden at editor startup")
    actors = [actor]
    if propagate:
        actors += list(actor.get_attached_actors(reset_array=True, recursively_include_attached_actors=True))
    hidden = not visible
    editor, game = scope in ("all", "editor"), scope in ("all", "game")
    with unreal.ScopedEditorTransaction("simul: set actor visibility"):
        for target in actors:
            target.modify()
        if editor and persistent:
            set_hidden_at_startup(actors, hidden)
        for target in actors:
            if editor:
                target.set_is_temporarily_hidden_in_editor(hidden)
            if game:
                target.set_actor_hidden_in_game(hidden)
    states = [flags(target) for target in actors]
    missed = [state["actor_path"] for state in states
              if (editor and state["hidden_in_editor"] != hidden)
              or (game and state["hidden_in_game"] != hidden)
              or (editor and persistent and state["hidden_in_editor_at_startup"] != hidden)]
    result = {"actor_path": actor_path, "visible": visible, "scope": scope, "propagate": propagate,
              "persistent": persistent, "actors": states}
    if missed:
        expected = {}
        if editor:
            expected["hidden_in_editor"] = hidden
            if persistent:
                expected["hidden_in_editor_at_startup"] = hidden
        if game:
            expected["hidden_in_game"] = hidden
        result.update(not_applied(_VISIBILITY_NOT_APPLIED, "Unreal did not apply the visibility to: "
                                  + ", ".join(missed), expected,
                                  [state for state in states if state["actor_path"] in missed]))
    return result

print(json.dumps(set_visibility(**args)))
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


def in_function(body: str) -> str:
    """``body`` run as the body of a function that is called once.

    A name a script binds at module level (a world, an actor, a render
    target) lives as long as the namespace the script ran in, which for a
    script run in the editor's shared Python namespace outlasts the call; a
    world held that way is a leak Unreal reports, or asserts on, when the map
    changes. Inside a function the names go when it returns. ``body`` holds no
    multi-line string literal, which the indentation would change.
    """
    return "def _simul_run():\n" + textwrap.indent(body.strip("\n"), "    ") + "\n\n_simul_run()\n"


# capture_lock(): the one HighResShot lock of the editor, a module in the
# editor's sys.modules, which outlives every script's namespace. ``owner`` is
# the token of the capture holding it, or None, and ``expires`` the time.time()
# it lapses at unless its owner renews it. The editor runs one script at a time
# on its game thread, so a script tests and takes it in one step.
CAPTURE_LOCK_HELPERS = """
import sys, time, types

def capture_lock():
    lock = sys.modules.get("_simul_capture_lock")
    if lock is None:
        lock = types.ModuleType("_simul_capture_lock")
        lock.owner, lock.expires = None, 0.0
        sys.modules["_simul_capture_lock"] = lock
    return lock
"""

#: Most child actor paths a listing entry names; its ``child_count`` counts
#: them all, and ``get_unreal_actor_info`` names more.
LISTED_CHILDREN = 20

#: Most child actor paths ``get_actor_info`` names; ``child_count`` counts them all.
INFO_CHILDREN = 200

# Helpers of the paged scene reads, after ACTOR_HELPERS (which holds EDITOR_WORLD).
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
LIST_ACTORS = ACTOR_HELPERS + SCENE_READ_HELPERS + in_function("""
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
""")

# Expects ``args`` = {"actor_path", "listed_children"}: actor_info with the
# attachment hierarchy (hierarchy) and pie_active.
ACTOR_INFO = ACTOR_HELPERS + SCENE_READ_HELPERS + in_function("""
actor = actor_at(args["actor_path"])
print(json.dumps(dict(actor_info(actor), **hierarchy(actor, args["listed_children"]), pie_active=pie_active())))
""")


def regex_searcher_helpers() -> str:
    """Source defining ``simul_regex_guard()``: ``simul.utils.regex_guard`` as a module in the editor.

    The module's own source, so the editor searches a tag exactly as the
    session checked the pattern and as Blender's add-on searches a name
    (``name_searcher``: ``re`` when it bounds the search, else a Thompson-NFA
    simulation linear in the tag). It is installed in the editor's
    ``sys.modules`` under a name with the source's digest, once per editor
    process and source, and needs only the standard library.
    """
    global _REGEX_SEARCHER_HELPERS
    if _REGEX_SEARCHER_HELPERS is None:
        from ..utils import regex_guard

        source = Path(regex_guard.__file__).read_text(encoding="utf-8")
        name = "_simul_regex_guard_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        _REGEX_SEARCHER_HELPERS = f"""
import sys, types

def simul_regex_guard():
    module = sys.modules.get({name!r})
    if type(module) is not types.ModuleType:
        module = types.ModuleType({name!r})
        sys.modules[{name!r}] = module
        try:
            exec(compile({source!r}, "<simul regex_guard>", "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop({name!r}, None)
            raise
    return module
"""
    return _REGEX_SEARCHER_HELPERS


_REGEX_SEARCHER_HELPERS: Optional[str] = None

# Expects ``args`` = {"label", "match_mode", "tag_length", "offset",
# "max_results", "budget", "listed_children"}, after regex_searcher_helpers in
# regex mode. Semantic labels are actor tags. Matching
# ignores case, as Unreal's tag comparison (FName) does: "exact" compares whole
# tags, "contains" looks for the label inside a tag and "regex" searches each
# tag for the regular expression. Every match is counted; one page of them,
# from ``offset``, is returned as listing entries (page). While PIE runs the
# editor world is searched (level_actors) and pie_active says so.
FIND_ACTORS_BY_TAG = ACTOR_HELPERS + SCENE_READ_HELPERS + in_function("""
label, mode = args["label"], args["match_mode"]
if mode == "regex":
    # regex_searcher_helpers' module: re when it bounds the search, else a
    # search linear in the tag, as Blender's add-on searches object names.
    search = simul_regex_guard().name_searcher(label, re.IGNORECASE, args["tag_length"], subject="label")

def matches(tag):
    if mode == "regex":
        return search(tag)
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
""")

#: Most components a scene graph node lists with ``include_components``; its
#: ``component_count`` counts them all, and ``list_unreal_actor_components``
#: pages through them.
GRAPH_COMPONENTS = 50

# Expects ``args`` = {"root_path", "max_depth", "include_components",
# "listed_components", "class_filter", "include_subclasses", "offset",
# "max_results", "budget"}.
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
# components that are not scene components), the first listed_components of
# them, with component_count counting them all and components_truncated when
# that is more (an actor with 1,502 components made a 160 KB page). While PIE
# runs the editor world is read (level_actors) and pie_active says so.
SCENE_GRAPH = ACTOR_HELPERS + SCENE_READ_HELPERS + in_function("""
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
    found = list(actor.get_components_by_class(unreal.ActorComponent))
    listed_components = []
    for component in found[:args["listed_components"]]:
        attach = component.get_attach_parent() if isinstance(component, unreal.SceneComponent) else None
        listed_components.append({
            "name": component.get_name(), "class_name": component.get_class().get_name(),
            "is_root": component == root,
            "attach_parent": attach.get_name() if attach is not None and component != root else None})
    return listed_components, len(found)

def node(actor, depth, max_depth):
    below = kept_children(actor)
    entry = {"name": actor.get_actor_label(), "path": actor.get_path_name(),
             "class_name": actor.get_class().get_name(), "child_count": len(below), "children": []}
    if filtering:
        entry["matches_filter"] = matches_class(actor.get_class())
    if args["include_components"]:
        entry["components"], entry["component_count"] = components_of(actor)
        if entry["component_count"] > len(entry["components"]):
            entry["components_truncated"] = True
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
""")

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
ACTOR_COMPONENTS = ACTOR_HELPERS + SCENE_READ_HELPERS + in_function("""
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
""")

# Scene digest in one round trip. Class keys are UClass path names
# (``/Script/Engine.StaticMeshActor``), the same strings Remote Control's
# ``/remote/object/describe`` reports as ``Class``. Static meshes, lights and
# cameras are counted by class, subclasses included: a light is a Light actor
# (directional, point, spot, rect) or a SkyLight, which is not a Light; a
# camera is a CameraActor (CineCameraActor included). A class name containing
# "Light" or "Camera" (LightmassImportanceVolume, CameraBlockingVolume) is
# neither.
SCENE_SUMMARY = EDITOR_WORLD + WORLD_PARTITION_HELPERS + in_function("""
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
    **world_partition_state(),
}))
""")

# Defines material_at(path, editor_library): the Material or Material Instance
# at ``path``, loaded with EditorAssetLibrary (which loads nothing during PIE)
# when ``editor_library`` is true, else with unreal.load_asset. The asset
# registry is asked first, so a path that names a level (a map package) is
# never loaded: loading it as an asset loads that map. Returns (material,
# reason): (None, None) when the path names nothing, and a reason naming what
# the path is when it is not a material. The library function is called
# inline, never passed around (see the execute_unreal_script hazard in
# docs/tools.md).
MATERIAL_AT = """
def material_at(path, editor_library):
    package = path.split(".", 1)[0]
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    kinds = [str(asset.asset_class_path.asset_name) for asset in registry.get_assets_by_package_name(package)]
    if "World" in kinds:
        return None, path + " is a level, not a material"
    asset = unreal.EditorAssetLibrary.load_asset(path) if editor_library else unreal.load_asset(path)
    if asset is None:
        return None, None
    if not isinstance(asset, unreal.MaterialInterface):
        return None, path + " is a " + asset.get_class().get_name() + ", not a material"
    return asset, None
"""

# Expects ``args`` = {"parent", "name", "folder"}. EditorAssetLibrary loads the
# parent, checks the destination and saves the instance, and while PIE runs it
# loads, finds and saves no assets (observed on UE 5.7.4: load_asset returns
# None and does_asset_exist False for assets that exist, and the editor logs
# "The Editor is currently in a play mode"). So the script prints the PIE
# refusal then, before anything is created, rather than a parent not found.
# Each failure is typed: no parent (NotFoundError), a parent that is not a
# material (InvalidAssetType: a StaticMesh used to be reported as "Parent
# material not found"), an existing destination (InvalidState), and an
# instance Unreal created but did not save (EngineError, naming it: the result
# of save_loaded_asset used to be ignored).
CREATE_MATERIAL_INSTANCE = EDITOR_WORLD + MATERIAL_AT + """
def create(parent, name, folder):
    if pie_active():
        return pie_refusal("create material instances")
    material, reason = material_at(parent, True)
    destination = folder + "/" + name
    if reason is not None:
        return {"success": False, "error_type": "InvalidAssetType",
                "error": reason + "; pass a Material or Material Instance as parent_path"}
    if material is None:
        return {"success": False, "error_type": "NotFoundError",
                "error": "No parent material at " + repr(parent) + ": pass a Material or Material Instance "
                         "asset path such as /Engine/BasicShapes/BasicShapeMaterial"}
    if unreal.EditorAssetLibrary.does_asset_exist(destination):
        return {"success": False, "error_type": "InvalidState",
                "error": "Asset already exists: " + destination + "; pass another instance_name or save_path"}
    # The factory's InitialParent is not exposed to Python; parent it after.
    instance = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, folder, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew(),
    )
    if instance is None:
        return {"success": False, "error_type": "EngineError",
                "error": "Unreal could not create a material instance at " + destination}
    unreal.MaterialEditingLibrary.set_material_instance_parent(instance, material)
    result = {"instance_path": instance.get_path_name(), "parent_path": material.get_path_name(),
              "class_name": instance.get_class().get_name()}
    if not unreal.EditorAssetLibrary.save_loaded_asset(instance):
        result.update(success=False, error_type="EngineError",
                      error="Unreal created " + instance.get_path_name() + " but did not save it; see the "
                            "editor log, then save it with save_unreal_assets")
    return result

print(json.dumps(create(**args)))
"""

# Expects ``args`` = {"actor_path", "material_path", "slot_index",
# "slot_name", "component_name"}. Assigns the material as the Details panel
# does: inside an undoable transaction, after modify() records the component,
# which also marks the package that holds it (the level, or the actor's own
# package in a World Partition level) dirty, so saving keeps the change.
# MeshComponent.set_material alone, which the tool called before, records and
# dirties nothing: on UE 5.7.4 "save dirty packages" saved nothing, reloading
# the level reverted the material and Undo did not undo it. The actor must be
# in the level being edited (actor_at), and the material is loaded only
# when the asset registry lists it as something other than a level, so a map
# path never loads that map. The component is the one named, else the actor's
# mesh root component, else its only mesh component: an actor with several
# used to fail with a traceback. The slot is the one named (a material slot
# name, as the Details panel shows it), else slot_index. The slot is read back
# after the write. A path that names something other than a material (a level,
# a mesh) is InvalidAssetType, as in create_unreal_material_instance; a path
# that names nothing is NotFoundError.
ASSIGN_MATERIAL = ACTOR_HELPERS + MATERIAL_AT + """
def described(components):
    return ", ".join(c.get_name() + " (" + c.get_class().get_name() + ")" for c in components)

def pick_component(actor, component_name):
    components = list(actor.get_components_by_class(unreal.MeshComponent))
    if component_name:
        for component in components:
            if component_name in (component.get_name(), component.get_path_name()):
                return component, None
        return None, ("The actor has no mesh component named " + repr(component_name)
                      + ("; its mesh components: " + described(components) if components else ""))
    root = actor.root_component
    if isinstance(root, unreal.MeshComponent):
        return root, None
    if len(components) == 1:
        return components[0], None
    if not components:
        return None, "The actor has no mesh component: " + actor.get_path_name()
    return None, ("The actor has " + str(len(components)) + " mesh components and none is its root; pass "
                  "component_name, one of: " + described(components))

def path_of(material):
    return material.get_path_name() if material is not None else None

def assign(actor_path, material_path, slot_index, slot_name, component_name):
    actor = actor_at(actor_path)
    material, reason = material_at(material_path, False)
    if reason is not None:
        return {"success": False, "error_type": "InvalidAssetType",
                "error": reason + "; pass a Material or Material Instance as material_path"}
    if material is None:
        return {"success": False, "error_type": "NotFoundError",
                "error": "No material at " + repr(material_path) + ": pass a Material or Material Instance "
                         "asset path such as /Engine/BasicShapes/BasicShapeMaterial"}
    component, problem = pick_component(actor, component_name)
    if component is None:
        return {"success": False, "error_type": "ValidationError", "error": problem}
    names = [str(name) for name in component.get_material_slot_names()]
    slots = component.get_num_materials()
    if slot_name:
        slot_index = component.get_material_index(slot_name)
        if slot_index < 0:
            return {"success": False, "error_type": "ValidationError",
                    "error": (component.get_name() + " has no material slot named " + repr(slot_name)
                              + "; its slots: " + (", ".join(names) if names else "none"))}
    elif not 0 <= slot_index < slots:
        return {"success": False, "error_type": "ValidationError",
                "error": ("slot_index " + str(slot_index) + " is out of range: " + component.get_name()
                          + " has " + str(slots) + " material slot(s)"
                          + (" (" + ", ".join(names) + ")" if names else ""))}
    previous = path_of(component.get_material(slot_index))
    with unreal.ScopedEditorTransaction("simul: assign material"):
        component.modify()
        component.set_material(slot_index, material)
    applied = path_of(component.get_material(slot_index))
    result = {"actor_path": actor.get_path_name(), "component_path": component.get_path_name(),
              "component_name": component.get_name(), "material_path": material.get_path_name(),
              "slot_index": slot_index, "slot_name": names[slot_index] if slot_index < len(names) else None,
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
# ``light_color`` as [r, g, b, a] bytes, the ranges already checked by the
# session. The light is the actor's LightComponentBase, so a SkyLight's
# intensity, color and shadows are set too; an actor without one, and a value
# its class lacks (a directional light's attenuation radius, a SkyLight's
# temperature), refuse the whole request before anything is written
# (ValidationError with details.code NOT_A_LIGHT or PROPERTY_NOT_ON_LIGHT).
# Writes through set_editor_property (the edit path the Details panel uses), in
# one undoable transaction: Remote Control property writes and the Set*
# functions silently no-op on a Stationary light's attenuation radius. Reads
# every value back; when Unreal stored one differently (it turned an intensity
# of -5 into 0 while the reply said it was not applied), every value is put
# back and the transaction cancelled, so a failed call changes nothing
# (LightParamsNotApplied, details.expected and details.actual). With no values
# it only reads the light.
SET_LIGHT_PARAMS = ACTOR_HELPERS + """
READ_BACK = ("intensity", "light_color", "temperature", "use_temperature", "attenuation_radius", "cast_shadows")

def read(component, name):
    value = component.get_editor_property(name)
    if isinstance(value, unreal.Color):
        return [value.r, value.g, value.b, value.a]
    return value

def has(component, name):
    try:
        component.get_editor_property(name)
    except Exception:
        return False
    return True

def write(component, name, value):
    if name == "light_color":
        value = unreal.Color(r=value[0], g=value[1], b=value[2], a=value[3])
    component.set_editor_property(name, value)

def differs(name, wanted, value):
    if isinstance(wanted, float):
        return abs(value - wanted) > 1e-3 * max(1.0, abs(wanted))
    return value != wanted

def unit_of(component):
    if isinstance(component, unreal.DirectionalLightComponent):
        return "lux"
    if isinstance(component, unreal.SkyLightComponent):
        return "scale"
    if has(component, "intensity_units"):
        return component.get_editor_property("intensity_units").name.lower()
    return None

def set_light():
    actor = actor_at(args["actor_path"])
    lights = actor.get_components_by_class(unreal.LightComponentBase)
    root = actor.root_component
    if isinstance(root, unreal.LightComponentBase):
        component = root
    elif len(lights) == 1:
        component = lights[0]
    else:
        return {"success": False, "error_type": "ValidationError",
                "error": actor.get_actor_label() + " (" + actor.get_class().get_name() + ") has "
                         + ("no light component" if not lights else str(len(lights)) + " light components and none "
                            "at its root") + "; pass a light actor (PointLight, SpotLight, RectLight, "
                         "DirectionalLight or SkyLight)",
                "details": {"code": "NOT_A_LIGHT", "actor_path": args["actor_path"], "light_components": len(lights)}}
    values = args["values"]
    lacking = [name for name in values if not has(component, name)]
    if lacking:
        return {"success": False, "error_type": "ValidationError",
                "error": component.get_class().get_name() + " has no " + ", ".join(lacking)
                         + "; nothing was changed",
                "details": {"code": "PROPERTY_NOT_ON_LIGHT", "parameters": lacking,
                            "light_class": component.get_class().get_name()}}
    rejected = {}
    if values:
        before = {name: read(component, name) for name in values}
        with unreal.ScopedEditorTransaction("simul: set light params") as transaction:
            for name, value in values.items():
                write(component, name, value)
            for name, wanted in values.items():
                stored = read(component, name)
                if differs(name, wanted, stored):
                    rejected[name] = stored
            if rejected:
                for name in values:
                    write(component, name, before[name])
                transaction.cancel()
    current = {name: read(component, name) for name in READ_BACK if has(component, name)}
    result = {"component_path": component.get_path_name(), "light_class": component.get_class().get_name(),
              "intensity_unit": unit_of(component),
              "applied": {} if rejected else {name: current[name] for name in values},
              "current": current, "params_set": 0 if rejected else len(values), "warning": None}
    if values and pie_active():
        result["warning"] = ("Play In Editor is running: this changed the editor world's light, which the "
                             "running simulation does not see; it shows the next time the simulation starts.")
    if rejected:
        result.update(success=False, error_type="LightParamsNotApplied",
                      error="Unreal did not take " + ", ".join(
                          name + " (it stored " + repr(stored) + ")" for name, stored in sorted(rejected.items()))
                      + "; every value was put back as it was, so nothing was changed",
                      details={"code": "NOT_APPLIED", "expected": {name: values[name] for name in rejected},
                               "actual": rejected})
    return result

print(json.dumps(set_light()))
"""

#: ``error_type`` of a physics edit refused because the body cannot take it:
#: ``details.problems`` lists each reason as {"code", "message"}.
INCOMPATIBLE_BODY = "IncompatibleBody"

# Shared by the physics and collision scripts below. Each edits one primitive
# component (the one named component_name, else the actor's root, else its
# only one, as physics_component picks).
# An actor in the editor world is edited the way the Details panel does it:
# inside one ScopedEditorTransaction, after modify(), so the edit is one Undo
# step and marks the map dirty, and Save All writes it. The runtime setters
# alone (set_simulate_physics and the like) change the live body but neither
# record an undo step nor dirty the map, so the edit was lost on exit (observed
# on UE 5.7.4); after modify() they are recorded, Undo and Redo restore them,
# the attachment a simulating body drops included, and they survive a save and
# reload (UE 5.7.4). The simulate flag is the exception: in the editor world
# it is written as the Details panel writes it, as the body instance's
# property (write_simulate). set_simulate_physics on a registered component
# detaches it from its parent at once, inside the editor actor itself: a part
# of a multi-part actor left its root's tree, so moving the actor left the
# part behind, Unreal's USD exporter (which walks the root's tree) wrote the
# actor as an empty Xform, and turning physics off did not reattach it (UE
# 5.7.4). Written as a property, the editor actor keeps its hierarchy; when a
# simulation starts Unreal detaches the play copy's body from its parent and
# welds the parts attached under it that do not simulate into its body
# (welded_to), and the editor actor is unchanged after Stop (UE 5.7.4).
# Cancelling a transaction only drops its record and keeps
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
# while Play In Editor runs, so the mesh checks are skipped then. A generated
# DynamicMeshActor's simple collision is read with GeometryScript (when the
# GeometryScripting plugin is loaded): one without any never started
# simulating, and a hull written to it fixed that (UE 5.7.4).
#
# The collision mode alone does not make a body collide: Unreal makes two
# bodies collide only when each blocks the other's object type, and a preset
# keeps its responses when the mode is changed. OverlapAll, OverlapAllDynamic
# and Trigger overlap every object type, so with collision_enabled true on top
# (QUERY_AND_PHYSICS, preset Custom) a simulated cube still fell through the
# floor at about 2000 cm/s while the tools reported it simulated (UE 5.7.4).
# So a body must block at least one object type (blocked_object_types) and the
# remedy offered is a preset that blocks, unless the body's own responses
# already do (NoCollision's block every object type, so collision_enabled true
# is the fix there).
#
# A multi-part actor (a SceneComponent root with several primitive
# components) has no one body: physics_component takes component_name to pick
# one (editor-only components, such as the sprite an Actor spawned from the
# class carries, never count), and refuses the actor without it (ValidationError AMBIGUOUS_COMPONENT,
# the components listed), as assign_unreal_material does; it used to fail
# with a ScriptError traceback. An actor with no primitive component at all is
# an IncompatibleBody (NO_PRIMITIVE_COMPONENT). Scripts raise SimulRefusal
# with the envelope and return it through refusable().
#
# Welding: in a simulation Unreal welds a part that does not simulate into the
# body of the simulating part it is attached under (Unreal's GetRootWelded
# walks primitive parents only), so the two move as one rigid body. The welded
# part's own body is that body: is_simulating_physics, get_mass (the parts'
# masses added up), the velocities, add_impulse and the mass, damping and
# gravity setters all act on it (a mass of 5 kg set on a welded Arm became its
# Chassis's override, UE 5.7.4). Such a part reads physics_mode "welded" with
# welded_to naming that part: in a simulation's world as Unreal welded it, and
# in the editor world as it will when a simulation starts.
PHYSICS_HELPERS = ACTOR_HELPERS + SIMULATION_WORLD + f"_INCOMPATIBLE_BODY = {INCOMPATIBLE_BODY!r}\n" + """
PHYSICS_COLLISION = ("QUERY_AND_PHYSICS", "PHYSICS_ONLY")
# Unreal's object types, by the name its Details panel shows and its CollisionChannel.
OBJECT_TYPES = (("WorldStatic", "ECC_WORLD_STATIC"), ("WorldDynamic", "ECC_WORLD_DYNAMIC"), ("Pawn", "ECC_PAWN"),
                ("PhysicsBody", "ECC_PHYSICS_BODY"), ("Vehicle", "ECC_VEHICLE"), ("Destructible", "ECC_DESTRUCTIBLE"))
COMPONENTS_LISTED = 20

class SimulRefusal(Exception):
    def __init__(self, payload):
        Exception.__init__(self, payload["error"])
        self.payload = payload

def refusable(run):
    try:
        return run()
    except SimulRefusal as refusal:
        return refusal.payload

def listed_components(components):
    return [{"component_name": c.get_name(), "component_class": c.get_class().get_name(),
             "component_path": c.get_path_name()} for c in components[:COMPONENTS_LISTED]]

def body_components(actor):
    # Its primitive components that can have a body: not editor-only ones such as the
    # BillboardComponent sprite an Actor spawned from the class gets (UE 5.7.4).
    return [c for c in actor.get_components_by_class(unreal.PrimitiveComponent) if not editor_only(c)]

def unplayed(path):
    # A path with the play world's UEDPIE_<n>_ prefix taken off its package name.
    return re.sub("/" + _PIE_PACKAGE, "/", path, count=1)

def physics_component(actor, component_name=None):
    components = body_components(actor)
    names = ", ".join(c.get_name() + " (" + c.get_class().get_name() + ")" for c in components[:COMPONENTS_LISTED])
    if component_name:
        for component in components:
            if (component_name in (component.get_name(), component.get_path_name())
                    or unplayed(component_name) == unplayed(component.get_path_name())):
                return component
        raise SimulRefusal({"success": False, "error_type": "ValidationError",
                            "error": actor.get_actor_label() + " has no primitive component named "
                                     + repr(component_name) + ("; its primitive components: " + names
                                                               if components else "; it has none"),
                            "details": {"code": "COMPONENT_NOT_FOUND", "parameter": "component_name",
                                        "value": component_name, "components": listed_components(components)}})
    root = actor.root_component
    if isinstance(root, unreal.PrimitiveComponent):
        return root
    if len(components) == 1:
        return components[0]
    if not components:
        problem = {"code": "NO_PRIMITIVE_COMPONENT", "message": (
            "it has no primitive component (a mesh or a collision shape), so Unreal gives it no physics body")}
        raise SimulRefusal({"success": False, "error_type": _INCOMPATIBLE_BODY,
                            "error": actor.get_actor_label() + ": " + problem["message"],
                            "details": {"code": "INCOMPATIBLE_BODY", "problems": [problem],
                                        "actor_path": actor.get_path_name()}})
    raise SimulRefusal({"success": False, "error_type": "ValidationError",
                        "error": actor.get_actor_label() + " has " + str(len(components)) + " primitive components "
                                 "and none is its root, so it has no one body; pass component_name, one of: " + names,
                        "details": {"code": "AMBIGUOUS_COMPONENT", "parameter": "component_name",
                                    "components": listed_components(components),
                                    "hint": "Pass component_name to act on one part's body. A part that simulates is "
                                            "a body of its own, and the parts attached under it that do not "
                                            "simulate are welded into its body in the simulation, moving with it "
                                            "as one rigid body: to simulate the actor as one body, enable physics "
                                            "on the part the others are attached under."}})

def own_flag(component):
    return bool(component.get_editor_property("body_instance").get_editor_property("simulate_physics"))

def weld_root(component):
    # The nearest primitive part above this one, through primitive parents only, that simulates
    # itself: the part whose body Unreal welds this one into when it does not simulate.
    parent = component.get_attach_parent()
    while isinstance(parent, unreal.PrimitiveComponent):
        if own_flag(parent):
            return parent
        parent = parent.get_attach_parent()
    return None

def welded_to(component):
    # The part whose body this one is welded into: in a simulation's world when Unreal welded it
    # (its body, which is_simulating_physics reads, is then that part's), in the editor world when
    # it will be once a simulation starts. None for a part that simulates itself or is not welded.
    if own_flag(component) or component.get_collision_enabled().name not in PHYSICS_COLLISION:
        return None
    root = weld_root(component)
    if root is None or root.mobility.name != "MOVABLE" or component.mobility.name != "MOVABLE":
        return None
    owner = component.get_owner()
    if owner is not None and in_simulation(owner):
        return root if component.is_simulating_physics() else None
    return root if root.get_collision_enabled().name in PHYSICS_COLLISION else None

def welded_parts(component):
    # The parts of its actor welded into this part's body, by name.
    owner = component.get_owner()
    if owner is None or not own_flag(component):
        return []
    return [part.get_name() for part in body_components(owner)
            if part != component and welded_to(part) == component]

def write_simulate(component, simulate, simulation):
    # The play copy takes the runtime setter, which detaches an attached body from its parent as Unreal
    # does when a body starts simulating (the copy is gone when the simulation stops). The editor world
    # takes the flag the way the Details panel writes it, which keeps the editor actor's hierarchy
    # (see the comment above PHYSICS_HELPERS).
    if simulation:
        component.set_simulate_physics(simulate)
        return
    body = component.get_editor_property("body_instance")
    body.set_editor_property("simulate_physics", simulate)
    component.set_editor_property("body_instance", body)

def blocked_object_types(component):
    channels = unreal.CollisionChannel
    return [name for name, channel in OBJECT_TYPES
            if component.get_collision_response_to_channel(getattr(channels, channel)).name == "ECR_BLOCK"]

def collision_fix(simulate):
    preset = "PhysicsActor" if simulate else "BlockAllDynamic"
    return ("call set_unreal_collision with collision_preset " + preset + ", which blocks every object type "
            "(collision_enabled true alone keeps the current preset's responses)")

def physics_mode(component):
    if component.get_collision_enabled().name not in PHYSICS_COLLISION:
        return "none"
    if component.mobility.name != "MOVABLE":
        return "static"
    if own_flag(component):
        return "simulated" if component.is_simulating_physics() else "kinematic"
    return "welded" if welded_to(component) is not None else "kinematic"

def physics_report(component, parts=True):
    # parts: list welded_parts, which walks the whole actor; a read of many bodies leaves it out.
    body = component.get_editor_property("body_instance")
    parent = component.get_attach_parent()
    override = bool(body.get_editor_property("override_mass"))
    weld = welded_to(component)
    return {
        "component_path": component.get_path_name(),
        "physics_mode": physics_mode(component),
        "welded_to": weld.get_path_name() if weld is not None else None,
        "welded_parts": welded_parts(component) if parts else [],
        "simulate_physics": bool(body.get_editor_property("simulate_physics")),
        "mobility": component.mobility.name.title(),
        "collision_mode": component.get_collision_enabled().name,
        "collision_preset": str(component.get_collision_profile_name()),
        "blocked_object_types": blocked_object_types(component),
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

def dynamic_mesh_problems(component, body, simulate):
    # A DynamicMeshComponent's simple collision, read with GeometryScript; [] when it cannot be read.
    if not hasattr(unreal, "GeometryScript_Collision"):
        return []
    try:
        complex_as_simple = component.get_editor_property("collision_type").name == "CTF_USE_COMPLEX_AS_SIMPLE"
    except Exception:
        complex_as_simple = False
    if complex_as_simple:
        if not simulate:
            return []
        return [{"code": "COMPLEX_AS_SIMPLE", "message": (
            "its dynamic mesh uses its triangles as simple collision (Collision Type: Use Complex Collision As "
            "Simple), which Unreal cannot simulate: give it simple collision with compute_unreal_convex_hull "
            "(output collision) or decompose_unreal_convex_hull, or make the body kinematic")}]
    try:
        library = unreal.GeometryScript_Collision
        count = library.get_simple_collision_shape_count(library.get_simple_collision_from_component(component))
    except Exception:
        return []
    if count:
        return []
    return [{"code": "NO_SIMPLE_COLLISION", "message": (
        "its dynamic mesh has no simple collision shape, so " + body + " body would pass through everything "
        "(Unreal does not start simulating it): give it simple collision with compute_unreal_convex_hull "
        "(output collision) or decompose_unreal_convex_hull")}]

def physics_problems(component, simulate, make_movable, collision=None, blocked=None, preset=None):
    # collision, blocked and preset: the mode, blocked object types and name of a preset applied in the
    # same call; None reads the component's.
    problems = []
    body = "a simulated" if simulate else "a kinematic"
    mobility = component.mobility.name
    if mobility != "MOVABLE" and not make_movable:
        problems.append({"code": "NOT_MOVABLE", "message": (
            "its mobility is " + mobility.title() + ", and Unreal moves and simulates only Movable "
            "bodies: pass make_movable true (the default) to make it Movable")})
    collision = collision or component.get_collision_enabled().name
    blocked = blocked_object_types(component) if blocked is None else blocked
    preset = preset or str(component.get_collision_profile_name())
    if collision not in PHYSICS_COLLISION:
        fix = ("call set_unreal_collision with collision_enabled true (its responses block " + ", ".join(blocked)
               + "), or with collision_preset " + ("PhysicsActor" if simulate else "BlockAllDynamic")
               if blocked else collision_fix(simulate))
        problems.append({"code": "NO_PHYSICS_COLLISION", "message": (
            "its collision is " + collision + " (preset " + preset + "), and " + body + " body needs physics "
            "collision (QUERY_AND_PHYSICS or PHYSICS_ONLY): " + fix)})
    elif not blocked:
        effect = ("it would fall through the floor and pass through everything" if simulate
                  else "simulated bodies would pass through it")
        problems.append({"code": "NO_BLOCKING_RESPONSE", "message": (
            "its collision responses (preset " + preset + ") overlap or ignore every object type ("
            + ", ".join(name for name, _ in OBJECT_TYPES) + "), and Unreal makes two bodies collide only "
            "when each blocks the other, so " + effect + ": " + collision_fix(simulate))})
    if hasattr(unreal, "DynamicMeshComponent") and isinstance(component, unreal.DynamicMeshComponent):
        problems += dynamic_mesh_problems(component, body, simulate)
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
    return problems

def apply(component, mode, simulation):
    if mode != "off" and component.mobility.name != "MOVABLE":
        component.set_mobility(unreal.ComponentMobility.MOVABLE)
    write_simulate(component, mode == "simulate", simulation)

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

def body_before(component):
    return {"simulate": bool(component.get_editor_property("body_instance").get_editor_property("simulate_physics")),
            "mobility": component.mobility, "parent": component.get_attach_parent(),
            "socket": component.get_attach_socket_name()}

def restore(component, before, simulation):
    write_simulate(component, before["simulate"], simulation)
    if component.mobility != before["mobility"]:
        component.set_mobility(before["mobility"])
    parent = before["parent"]
    if parent is not None and component.get_attach_parent() != parent:
        rule = unreal.AttachmentRule.KEEP_WORLD
        component.attach_to_component(parent, before["socket"], rule, rule, rule, False)

def write_params(component, values):
    if "mass" in values:
        component.set_mass_override_in_kg("None", values["mass"], True)
    if "linear_damping" in values:
        component.set_linear_damping(values["linear_damping"])
    if "angular_damping" in values:
        component.set_angular_damping(values["angular_damping"])
    if "enable_gravity" in values:
        component.set_enable_gravity(values["enable_gravity"])

def preset_collision(name):
    # The collision mode a preset sets and the object types it blocks, read off a
    # transient component, or None for a name Unreal does not know (it renames
    # that to Custom).
    probe = unreal.new_object(unreal.StaticMeshComponent)
    probe.set_collision_profile_name(name)
    if str(probe.get_collision_profile_name()).lower() != name.lower():
        return None
    return probe.get_collision_enabled().name, blocked_object_types(probe)

def snapshot(component):
    # Everything a physics setup writes, to put back: the body instance (a copy of
    # the struct: collision preset, responses and mode, mass, damping, gravity and
    # the simulate flag), mobility, attachment and Use Default Collision. The
    # struct get_editor_property returns refers to the component's own, so later
    # writes showed through it and putting it back changed nothing (UE 5.7.4);
    # copy() takes it as it is now.
    snap = body_before(component)
    snap["body"] = component.get_editor_property("body_instance").copy()
    snap["default_collision"] = (bool(component.get_editor_property("use_default_collision"))
                                 if isinstance(component, unreal.StaticMeshComponent) else None)
    return snap

def put_back(component, snap):
    restore(component, snap, False)
    component.set_editor_property("body_instance", snap["body"])
    if snap["default_collision"] is not None:
        component.set_editor_property("use_default_collision", snap["default_collision"])

def expected_mode(mode):
    return {"simulate": "simulated", "kinematic": "kinematic"}.get(mode, "not simulating")

def refusal(actor, what, problems, component):
    part = "" if component == actor.root_component else " (component " + component.get_name() + ")"
    return {"success": False, "error_type": _INCOMPATIBLE_BODY,
            "error": "Cannot " + what + " " + actor.get_actor_label() + part + ": "
                     + "; ".join(problem["message"] for problem in problems),
            "details": {"code": "INCOMPATIBLE_BODY", "problems": problems,
                        "component_path": component.get_path_name()}}

def blocking_note(report):
    # What a body with physics collision that blocks some object types, but not the level's floors, does.
    blocked = report["blocked_object_types"]
    if report["collision_mode"] not in PHYSICS_COLLISION or not blocked or "WorldStatic" in blocked:
        return None
    return ("The body blocks only " + ", ".join(blocked) + ", not WorldStatic, the object type of the level's "
            "static floors and walls (StaticMeshActors and the like), so it passes through them; collision_preset "
            "PhysicsActor blocks every object type.")

def edit_note(simulation):
    if simulation:
        return ("This changed the running simulation's copy of the actor only: Unreal records no undo "
                "for it and it is gone when the simulation stops.")
    if pie_active():
        return ("Play In Editor is running: this changed the editor world, which the running "
                "simulation does not see; it takes effect the next time the simulation starts.")
    return None
"""

# Expects ``args`` = {"actor_path", "component_name" (or None), "mode"
# ("simulate", "kinematic" or "off"), "make_movable", "transaction"}.
# "simulate" makes the body
# simulate physics, "kinematic" keeps its physics collision without simulating
# (simulated bodies hit it; it moves only when its transform is set) and "off"
# stops it simulating, collision and mobility kept. A body that cannot take
# the mode is refused with nothing written (IncompatibleBody). A Static or
# Stationary body is made Movable when make_movable is true. In the editor
# world the editor actor keeps its hierarchy (write_simulate); the reply says
# what the simulation does with an attached body (simulation_detaches_from)
# and which parts it welds into it (welded_parts). In the play copy Unreal
# detaches an attached body as it starts simulating (detached_from). If the
# body still does not read back as asked, every value is restored and the
# transaction cancelled.
ENABLE_PHYSICS = PHYSICS_HELPERS + """
def simulation_notes(component, mode, before, weld):
    # What the simulation does with this editor-world body: Unreal detaches an attached body from its
    # parent in the play copy as it starts simulating, and welds into it the parts attached under it.
    notes, parent = [], before["parent"]
    if mode == "simulate" and parent is not None:
        if weld is not None:
            notes.append("In the simulation Unreal detaches " + component.get_name() + " from " + parent.get_name()
                         + " as it starts simulating, so it becomes a body of its own and no longer moves with "
                         + weld.get_name() + ", whose body it was welded into; with its physics off it stays "
                         "welded into " + weld.get_name() + "'s body. The editor actor keeps its hierarchy.")
        else:
            notes.append("In the simulation Unreal detaches " + component.get_name() + " from "
                         + parent.get_path_name() + " as it starts simulating, so the body moves while "
                         + parent.get_name() + " stays where the simulation started; the editor actor keeps "
                         "its hierarchy.")
    parts = welded_parts(component)
    if parts:
        notes.append(", ".join(parts) + (" is" if len(parts) == 1 else " are") + " attached under it without "
                     "simulating, so in the simulation Unreal welds " + ("it" if len(parts) == 1 else "them")
                     + " into its body: they move as one rigid body, of their masses added up.")
    return notes

def enable():
    actor = actor_at(args["actor_path"], play_world=True)
    component = physics_component(actor, args["component_name"])
    mode = args["mode"]
    if mode != "off":
        problems = physics_problems(component, mode == "simulate", args["make_movable"])
        if problems:
            what = "simulate physics on" if mode == "simulate" else "make kinematic"
            return refusal(actor, what, problems, component)
    before = body_before(component)
    weld = welded_to(component)
    simulation = in_simulation(actor)
    failure = None
    if simulation:
        apply(component, mode, True)
        failure = unmet(component, mode)
        if failure is not None:
            restore(component, before, True)
    else:
        with unreal.ScopedEditorTransaction(args["transaction"]) as transaction:
            component.modify()
            try:
                apply(component, mode, False)
                failure = unmet(component, mode)
            except Exception as exc:
                failure = str(exc) or type(exc).__name__
            if failure is not None:
                restore(component, before, False)
                transaction.cancel()
    if failure is not None:
        report = physics_report(component)
        return {"success": False, "error_type": "PhysicsNotApplied",
                "error": failure + "; nothing was changed",
                "details": dict(report, code="NOT_APPLIED", expected={"physics_mode": expected_mode(mode)},
                                actual={"physics_mode": report["physics_mode"],
                                        "simulate_physics": report["simulate_physics"]})}
    result = physics_report(component)
    parent = before["parent"]
    result.update(actor_path=actor.get_path_name(), world="simulation" if simulation else "editor",
                  physics_enabled=result["physics_mode"] == "simulated",
                  mobility_changed_from=None if component.mobility == before["mobility"]
                  else before["mobility"].name.title(),
                  detached_from=None, simulation_detaches_from=None)
    warnings = []
    if simulation:
        if parent is not None and component.get_attach_parent() is None:
            result["detached_from"] = parent.get_path_name()
            warnings.append("Unreal detached this play copy's " + component.get_name() + " from " + parent.get_name()
                            + " as it started simulating" + (", so it no longer moves with " + weld.get_name()
                                                             if weld is not None else "")
                            + "; the editor actor is unchanged.")
    else:
        if mode == "simulate" and parent is not None:
            result["simulation_detaches_from"] = parent.get_path_name()
        warnings += simulation_notes(component, mode, before, weld)
    if mode != "off":
        warnings.append(blocking_note(result))
    warnings.append(edit_note(simulation))
    result["warning"] = " ".join(warning for warning in warnings if warning) or None
    return result

print(json.dumps(refusable(enable)))
"""

# Expects ``args`` = {"actor_path", "component_name" (or None), "values":
# {"mass", "linear_damping", "angular_damping", "enable_gravity"} (only those
# given), "transaction"}.
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

def welded_refusal(actor, component, weld):
    names = ", ".join(sorted(args["values"]))
    problem = {"code": "WELDED", "message": (
        component.get_name() + " is welded into " + weld.get_name() + "'s body in the running simulation, so "
        "Unreal moves them as one body and writes " + names + " to that body, " + weld.get_name() + "'s own "
        "values, not to " + component.get_name() + ": set them on " + weld.get_name() + " (component_name "
        + weld.get_name() + "), or on the editor actor's " + component.get_name() + ", whose mass then adds to "
        "the welded body the next time the simulation starts")}
    return {"success": False, "error_type": _INCOMPATIBLE_BODY,
            "error": "Cannot set " + names + " on " + actor.get_actor_label() + " (component "
                     + component.get_name() + "): " + problem["message"] + "; nothing was changed",
            "details": {"code": "WELDED", "problems": [problem], "component_path": component.get_path_name(),
                        "welded_to": weld.get_path_name()}}

def set_params():
    actor = actor_at(args["actor_path"], play_world=True)
    component = physics_component(actor, args["component_name"])
    values = args["values"]
    simulation = in_simulation(actor)
    weld = welded_to(component)
    if values and weld is not None and (simulation or component.is_simulating_physics()):
        # Welded now: the setters would write the weld parent's body (see PHYSICS_HELPERS).
        return welded_refusal(actor, component, weld)
    if values:
        if simulation:
            write_params(component, values)
        else:
            with unreal.ScopedEditorTransaction(args["transaction"]):
                component.modify()
                write_params(component, values)
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
    elif weld is not None and values:
        warnings.append("In the simulation " + component.get_name() + " is welded into " + weld.get_name()
                        + "'s body: they move as one body, whose mass is the parts' masses added up and whose "
                        "damping and gravity are " + weld.get_name() + "'s.")
    note = edit_note(simulation)
    if note and values:
        warnings.append(note)
    result["warning"] = " ".join(warnings) or None
    if rejected:
        result.update(success=False, error_type="PhysicsParamsNotApplied",
                      error="Unreal did not apply: " + ", ".join(rejected),
                      details={"code": "NOT_APPLIED", "expected": {name: values[name] for name in rejected},
                               "actual": {name: read[name] for name in rejected}})
    return result

print(json.dumps(refusable(set_params)))
"""

# Expects ``args`` = {"actor_path", "component_name" (or None), "preset" (or
# None), "enabled" (True, False or None), "transaction"}. A preset is checked on a transient component
# first, since Unreal renames an unknown preset to Custom instead of failing,
# so an unknown one changes nothing. Setting collision_enabled on a static mesh
# component clears its Use Default Collision flag first, as setting a preset
# does in Unreal (UStaticMeshComponent::SetCollisionProfileName): with the flag
# set, Unreal puts the mesh's own collision back when the component is
# registered again, in the play copy when the simulation starts and after the
# map reloads, so a collision turned off came back as BlockAll (UE 5.7.4).
# collision_enabled true keeps the preset's responses, so on an overlap preset
# (OverlapAll, Trigger) the body still blocks nothing; the reply's warning says
# so and names the preset that blocks (see PHYSICS_HELPERS).
SET_COLLISION = PHYSICS_HELPERS + """
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
    component = physics_component(actor, args["component_name"])
    preset, enabled = args["preset"], args["enabled"]
    if preset and preset_collision(preset) is None:
        return {"success": False, "error_type": "ValidationError",
                "error": "Unknown collision preset: " + preset + "; nothing was changed. Presets include "
                         "BlockAll, BlockAllDynamic, PhysicsActor, OverlapAll, NoCollision and Custom",
                "details": {"code": "UNKNOWN_PRESET", "parameter": "collision_preset", "value": preset}}
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
    elif result["collision_mode"] in PHYSICS_COLLISION and not result["blocked_object_types"]:
        warnings.append("Collision is " + result["collision_mode"] + ", but its responses (preset "
                        + result["collision_preset"] + ", kept from the preset it had) overlap or ignore every object "
                        "type, so nothing collides with this body: physics bodies pass through it"
                        + (", and as it simulates it falls through the floor" if result["simulate_physics"] else "")
                        + ". For a body that collides, set collision_preset PhysicsActor (a simulated body), "
                        "BlockAllDynamic (a Movable one moved by its transform, such as a kinematic body) or BlockAll "
                        "(a static one).")
    else:
        warnings.append(blocking_note(result) if result["simulate_physics"] else None)
    if preset or enabled is not None:
        warnings.append(edit_note(simulation))
    result["warning"] = " ".join(warning for warning in warnings if warning) or None
    return result

print(json.dumps(refusable(set_collision)))
"""

# Expects ``args`` = {"paths", "tag", "folder", "actor_filter" (each or None), "mode"
# ("simulate", "kinematic" or "off"), "make_movable", "preset"
# (or None), "values" (set_physics_params' checked values), "skip_failures",
# "limit", "listed", "transaction"}. Sets up physics on many actors in one
# call and one undoable transaction: the actors at paths, plus every editor
# actor with the tag, in the outliner folder (or one under it) and whose label
# or path contains actor_filter (ignoring case), each that is given. Each actor
# is checked as enable_physics checks one, the preset's collision standing in
# for its own when a preset is given, before anything is written; unless
# skip_failures, one that cannot be set up refuses the whole call. Each actor is
# snapshot before it is written (snapshot: the body instance struct with its
# preset, responses, mass, damping, gravity and simulate flag, plus mobility,
# attachment and Use Default Collision). The simulate flag is written as
# enable_physics writes it in the editor world, so every actor keeps its
# attachment; simulation_detaches_from names the parent an attached one leaves
# in the simulation. One that still does not read back as
# asked is put back whole; without skip_failures every actor already written is
# put back too and the transaction cancelled, so the call changes nothing
# (cancelling alone only drops the undo record). A call that sets up no actor
# fails. Paths in the running simulation's world are refused: this edits the
# editor world.
BULK_PHYSICS = PHYSICS_HELPERS + """
def in_folder(actor, folder):
    path = str(actor.get_folder_path())
    return path == folder or path.startswith(folder + "/")

def select():
    chosen, failed, seen = [], [], set()
    for path in args["paths"]:
        try:
            actor = actor_at(path, play_world=True)
        except SimulNotFound as exc:
            failed.append({"actor_path": path, "error": str(exc)})
            continue
        if in_simulation(actor):
            failed.append({"actor_path": path, "error": "is in the running simulation's world; bulk setup edits "
                                                        "the editor world, so pass the editor path"})
        elif actor.get_path_name() not in seen:
            chosen.append(actor)
            seen.add(actor.get_path_name())
    needle = (args["actor_filter"] or "").lower()
    if args["tag"] or args["folder"] or needle:
        for actor in level_actors():
            if args["tag"] and not actor.actor_has_tag(args["tag"]):
                continue
            if args["folder"] and not in_folder(actor, args["folder"]):
                continue
            if (needle and needle not in actor.get_actor_label().lower()
                    and needle not in actor.get_path_name().lower()):
                continue
            if actor.get_path_name() not in seen:
                chosen.append(actor)
                seen.add(actor.get_path_name())
    return chosen, failed

def setup():
    mode, preset = args["mode"], args["preset"]
    preset_mode = preset_blocked = None
    if preset:
        known = preset_collision(preset)
        if known is None:
            return {"success": False, "error_type": "ValidationError",
                    "error": "Unknown collision preset: " + preset + "; nothing was changed",
                    "details": {"code": "UNKNOWN_PRESET", "parameter": "collision_preset", "value": preset}}
        preset_mode, preset_blocked = known
    actors, failed = select()
    # The paths that found no actor of the editor world count as matched, and failed.
    matched = len(actors) + len(failed)
    if len(actors) > args["limit"]:
        return {"success": False, "error_type": "ValidationError",
                "error": str(len(actors)) + " actors matched, more than the " + str(args["limit"])
                         + " one call sets up; narrow the selection by tag, folder or paths",
                "details": {"code": "TOO_MANY_ACTORS", "matched": len(actors), "maximum": args["limit"]}}
    plan = []
    for actor in actors:
        try:
            component = physics_component(actor)
        except SimulRefusal as refused:
            details = refused.payload.get("details") or {}
            error = refused.payload["error"]
            if details.get("code") == "AMBIGUOUS_COMPONENT":
                error = (actor.get_actor_label() + " has several primitive components and none is its root, and "
                         "bulk setup sets up one body an actor: set up the part the others are attached under with "
                         "enable_unreal_physics and component_name (the parts attached under it that do not "
                         "simulate are welded into its body in the simulation); its parts: " + ", ".join(
                             c["component_name"] for c in details.get("components", [])))
            failed.append({"actor_path": actor.get_path_name(), "error": error,
                           "problems": [p["code"] for p in details.get("problems", [])] or [details.get("code")]})
            continue
        except Exception as exc:
            failed.append({"actor_path": actor.get_path_name(), "error": str(exc)})
            continue
        problems = [] if mode == "off" else physics_problems(
            component, mode == "simulate", args["make_movable"], preset_mode, preset_blocked, preset)
        if problems:
            failed.append({"actor_path": actor.get_path_name(), "problems": [p["code"] for p in problems],
                           "error": "; ".join(p["message"] for p in problems)})
            continue
        plan.append((actor, component))
    if failed and not args["skip_failures"]:
        return {"success": False, "error_type": _INCOMPATIBLE_BODY,
                "error": str(len(failed)) + " of the " + str(matched) + " actors cannot be set up, so none was "
                         "changed (pass skip_failures true to set up the rest). First: " + failed[0]["actor_path"]
                         + ": " + failed[0]["error"],
                "details": {"code": "INCOMPATIBLE_BODY", "failed": failed[:args["listed"]],
                            "failed_count": len(failed), "matched": matched}}
    if not plan and not failed:
        return {"success": False, "error_type": "ValidationError",
                "error": "No actor matched the paths, tag, folder and actor_filter given; nothing was changed",
                "details": {"code": "NO_MATCH"}}
    configured, detaching, written, not_applied = [], 0, [], {}
    expected = {"physics_mode": expected_mode(mode)}
    if preset:
        expected["collision_preset"] = preset
    expected.update(args["values"])
    if plan:
        with unreal.ScopedEditorTransaction(args["transaction"]) as transaction:
            for actor, component in plan:
                before = snapshot(component)
                component.modify()
                if preset:
                    component.set_collision_profile_name(preset)
                apply(component, mode, False)
                write_params(component, args["values"])
                problem = unmet(component, mode)
                if problem is not None:
                    report = physics_report(component)
                    not_applied[actor.get_path_name()] = {"physics_mode": report["physics_mode"],
                                                          "simulate_physics": report["simulate_physics"],
                                                          "mobility": report["mobility"]}
                    put_back(component, before)
                    failed.append({"actor_path": actor.get_path_name(),
                                   "error": problem + "; it was put back as it was"})
                    if not args["skip_failures"]:
                        for done_component, done_before in reversed(written):
                            put_back(done_component, done_before)
                        transaction.cancel()
                        return {"success": False, "error_type": "PhysicsNotApplied",
                                "error": actor.get_actor_label() + ": " + problem + "; every actor was put back "
                                         "as it was, so nothing was changed (pass skip_failures true to keep the "
                                         "actors that do take it)",
                                "details": {"code": "NOT_APPLIED", "expected": expected, "actual": not_applied,
                                            "failed": failed[:args["listed"]], "failed_count": len(failed),
                                            "matched": matched}}
                    continue
                written.append((component, before))
                report = physics_report(component)
                leaves = (before["parent"].get_path_name()
                          if mode == "simulate" and before["parent"] is not None else None)
                detaching += leaves is not None
                configured.append({"actor_path": actor.get_path_name(), "physics_mode": report["physics_mode"],
                                   "mobility_changed_from": None if component.mobility == before["mobility"]
                                   else before["mobility"].name.title(),
                                   "simulation_detaches_from": leaves, "mass_kg": report["mass_kg"]})
    warnings = []
    if detaching:
        warnings.append(str(detaching) + " of the bodies set up to simulate " + ("is" if detaching == 1 else "are")
                        + " attached to a parent: in the simulation Unreal detaches each from it as it starts "
                        "simulating (simulation_detaches_from); the editor actors keep their attachment.")
    if pie_active():
        warnings.append("Play In Editor is running: this changed the editor world, which the running simulation "
                        "does not see; it takes effect the next time the simulation starts.")
    result = {"world": "editor", "mode": mode, "matched": matched, "configured_count": len(configured),
              "failed_count": len(failed), "configured": configured[:args["listed"]],
              "failed": failed[:args["listed"]], "warning": " ".join(warnings) or None}
    if not configured:
        result.update(success=False, error_type=_INCOMPATIBLE_BODY,
                      error="None of the " + str(matched) + " actors could be set up; nothing was changed",
                      details={"code": "NOTHING_SET_UP", "expected": expected, "actual": not_applied,
                               "failed": failed[:args["listed"]], "failed_count": len(failed)})
    return result

print(json.dumps(setup()))
"""

#: Name of the editor-side module that keeps continuous forces going. Versioned
#: so a newer simul never runs an older editor session's tick function.
FORCE_MODULE = "_simul_unreal_forces_v2"

# The longest physics step Unreal takes in one frame, as UWorld::SetupPhysicsTickFunctions
# hands it to FChaosScene::SetUpForFrame (UE 5.7): min(frame time, MaxPhysicsDeltaTime),
# 1/30 s by default, or with substepping min(frame time, MaxSubsteps x
# MaxSubstepDeltaTime). A frame longer than that advances game time by the whole
# frame (up to 0.4 s, MaxDeltaTime) but physics, gravity included, by the limit
# only. None means no limit. PhysicsSettings is read through its default
# object; where Python cannot read it the engine default, 1/30 s, stands.
PHYSICS_STEP_LIMIT = """
def physics_step_limit():
    try:
        settings = unreal.PhysicsSettings.get_default_object()
        if settings.get_editor_property("substepping"):
            step = settings.get_editor_property("max_substep_delta_time")
            return step * settings.get_editor_property("max_substeps") if step > 0 else None
        limit = settings.get_editor_property("max_physics_delta_time")
        return limit if limit > 0 else None
    except Exception:
        return 1.0 / 30.0
"""

# The source of FORCE_MODULE. A force added once acts for one physics step:
# Unreal applies AddForce to the next step and then clears it, so one call at
# twice the body's weight gave a 19 cm/s blip (UE 5.7.4). A continuous force is a
# job that a ticker (unreal.register_ticker_callback, called once every editor
# frame, after the world has ticked, headless too) serves each frame: it adds,
# as an impulse, the momentum the force carries over the physics time that frame
# simulated, the frame's game time capped at the physics step limit
# (PHYSICS_STEP_LIMIT), and counts the duration in that physics time. So the
# force keeps pace with gravity: counting game time instead gave a body held up
# by exactly its weight +1216 cm/s at 10 fps, as each 0.1 s frame added 0.1 s of
# force but only 1/30 s of gravity. Remote Control runs the request, and the
# ticker runs, after the frame's world tick, so an impulse lands in the next
# frame's step: APPLY_FORCE delivers the first step's share in the request
# itself (sized by the frame just simulated), and each ticker call delivers the
# next one, sized by the step that frame took. So no step goes uncompensated,
# and a frame whose length changes shifts the push by at most that change.
# Paused time adds nothing. The job names the component by path and
# finds it each frame, so it holds no reference to the play world; it ends when
# the simulation stops or restarts (its game time runs behind the job's) or the
# body is destroyed, and the ticker removes itself when no job is left
# (returning False removes a ticker callback).
_FORCE_MODULE_SOURCE = """
import unreal

jobs = []
handle = None

def serve(job, world):
    component = unreal.find_object(None, job["component"])
    if component is None or not unreal.SystemLibrary.is_valid(component):
        return False
    now = unreal.GameplayStatics.get_time_seconds(world)
    if now < job["last"]:
        return False
    elapsed = now - job["last"]
    job["last"] = now
    step = elapsed if job["limit"] is None else min(elapsed, job["limit"])
    step = min(step, job["duration"] - job["served"])
    if step > 0:
        impulse = unreal.Vector(*[value * step for value in job["force"]])
        if job["location"] is None:
            component.add_impulse(impulse)
        else:
            component.add_impulse_at_location(impulse, unreal.Vector(*job["location"]))
        job["served"] += step
        job["frames"] += 1
    return job["served"] < job["duration"] - 1e-9

def tick(delta_seconds):
    global handle
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
    jobs[:] = [job for job in jobs
               if world is not None and world.get_path_name() == job["world"] and serve(job, world)]
    if jobs:
        return True
    if handle is not None:
        try:
            unreal.unregister_ticker_callback(handle)
        except Exception:
            pass
    handle = None
    return False

def start(job):
    global handle
    jobs.append(job)
    if handle is None:
        handle = unreal.register_ticker_callback(tick)
"""

# Expects ``args`` = {"actor_path", "component_name" (or None), "vector" [x, y,
# z] in Unreal units (kg cm/s^2 for a force, kg cm/s for an impulse),
# "location" (world cm) or None, "impulse", "duration" (s of physics time; 0 is
# one physics step)}. component_name picks the body of a multi-part actor; the
# copy's component of that name is pushed. A part welded into another's body
# (physics_mode "welded") is pushed too: Unreal's AddImpulse acts on the welded
# body, which moved as one (an impulse on a welded Arm gave its Chassis and
# Wheel the same velocity, UE 5.7.4), and the reply names it in welded_to. A
# part that neither simulates nor is welded is refused; when it is attached
# under another primitive part the refusal names that part, whose body would
# weld it in, rather than the part itself, which would leave its parent in
# the simulation. Acts on the
# actor's copy in the running simulation (SIMULATION_WORLD), the only world
# whose bodies step; with no simulation running it refuses (SimulationNotRunning).
APPLY_FORCE = (
    PHYSICS_HELPERS
    + PHYSICS_STEP_LIMIT
    + f"_FORCE_MODULE, _FORCE_MODULE_SOURCE = {FORCE_MODULE!r}, {_FORCE_MODULE_SOURCE!r}\n"
    + """
import sys, types

def force_module():
    module = sys.modules.get(_FORCE_MODULE)
    if module is None:
        module = types.ModuleType(_FORCE_MODULE)
        exec(_FORCE_MODULE_SOURCE, module.__dict__)
        sys.modules[_FORCE_MODULE] = module
    return module

def push(component, vector, location, impulse):
    if impulse:
        if location is None:
            component.add_impulse(unreal.Vector(*vector))
        else:
            component.add_impulse_at_location(unreal.Vector(*vector), unreal.Vector(*location))
    elif location is None:
        component.add_force(unreal.Vector(*vector))
    else:
        component.add_force_at_location(unreal.Vector(*vector), unreal.Vector(*location))

def apply():
    actor = actor_at(args["actor_path"], play_world=True)
    world = simulation_world()
    if world is None:
        return simulation_refusal("apply_unreal_force", "the editor world's bodies never step, so a force "
                                  "applied there would be lost")
    copy = simulation_counterpart(actor)
    if copy is None:
        return {"success": False, "error_type": _NOT_IN_SIMULATION,
                "error": actor.get_actor_label() + " has no live copy in the running simulation ("
                         + world.get_path_name() + "): " + _NO_COPY_REASONS + "; list_unreal_simulation_actors "
                         "lists the simulation's actors",
                "details": {"code": "NOT_IN_SIMULATION", "actor_path": args["actor_path"],
                            "hint": "For an actor of a World Partition level, call focus_unreal_on_actor on it and "
                                    "retry; for one added after the simulation started, stop and start it again."}}
    name = args["component_name"]
    if name and "/" in name:
        # A component path of the editor actor names the copy's component of the same name, wherever
        # World Partition streamed the copy (a cell level's path is not the editor's with a prefix).
        for source in (actor, editor_counterpart(copy)):
            if source is not None and source != copy:
                name = next((c.get_name() for c in body_components(source) if c.get_path_name() == name), name)
    component = physics_component(copy, name)
    mode = physics_mode(component)
    weld = welded_to(component) if mode == "welded" else None
    if mode not in ("simulated", "welded"):
        part = "" if component == copy.root_component else " (component " + component.get_name() + ")"
        carrier, parent = None, component.get_attach_parent()
        while isinstance(parent, unreal.PrimitiveComponent):
            carrier, parent = parent, parent.get_attach_parent()
        if carrier is not None:
            # A part attached under another: simulating it alone would detach it from that part in the
            # simulation, so name the part that carries it, whose body welds it in.
            fix = ("enable_unreal_physics with component_name " + carrier.get_name() + " on the editor actor and "
                   "restart the simulation: " + carrier.get_name() + " then simulates with the parts attached "
                   "under it, this one included, welded into its body, and a force on any of them moves that "
                   "body (enabling physics on " + component.get_name() + " itself would detach it from "
                   + component.get_attach_parent().get_name() + " in the simulation, so it would move on its own)")
        else:
            fix = ("enable_unreal_physics on the editor actor and restart the simulation, or on the simulation "
                   "path for this run")
        return {"success": False, "error_type": _INCOMPATIBLE_BODY,
                "error": copy.get_actor_label() + part + " does not simulate physics in the running simulation "
                         "(physics_mode " + mode + "), so a force moves nothing: " + fix,
                "details": dict(physics_report(component), code="NOT_SIMULATING")}
    vector, location = args["vector"], args["location"]
    paused = bool(unreal.GameplayStatics.is_game_paused(world))
    now = unreal.GameplayStatics.get_time_seconds(world)
    limit = physics_step_limit()
    if args["impulse"]:
        push(component, vector, location, True)
        application = "impulse"
    elif args["duration"] > 0:
        # The first step's share now, so it lands with the next step's gravity (see _FORCE_MODULE_SOURCE).
        frame = unreal.GameplayStatics.get_world_delta_seconds(world)
        first = 0.0 if paused else min(frame if limit is None else min(frame, limit), args["duration"])
        if first > 0:
            push(component, [value * first for value in vector], location, True)
        if first < args["duration"] - 1e-9:
            force_module().start({"world": world.get_path_name(), "component": component.get_path_name(),
                                  "force": list(vector), "location": location, "last": now, "limit": limit,
                                  "duration": args["duration"], "served": first, "frames": 1 if first else 0})
        application = "continuous"
    else:
        push(component, vector, location, False)
        application = "one_frame"
    mass = component.get_mass()
    seconds = None if application == "one_frame" else (1.0 if application == "impulse" else args["duration"])
    expected = [value * seconds / mass for value in vector] if seconds is not None and mass > 0 else None
    warnings = []
    if weld is not None:
        warnings.append(component.get_name() + " is welded into " + weld.get_name() + "'s body, so the force acts on "
                        "that whole body, every part welded into it moving as one: mass_kg is its mass, the parts' "
                        "masses added up.")
    if application == "one_frame":
        frame = unreal.GameplayStatics.get_world_delta_seconds(world)
        step = frame if limit is None else min(frame, limit)
        warnings.append("A force with duration 0 acts for one physics step (the frame time, at most the "
                        "physics step limit; about " + format(step, ".4f") + " s now), changing the velocity "
                        "by about force x step / mass; pass duration for a sustained force, or is_impulse for "
                        "an instant change.")
    if paused:
        warnings.append("The simulation is paused: the force acts once it resumes.")
    return {"actor_path": args["actor_path"], "simulation_actor_path": copy.get_path_name(),
            "component_path": component.get_path_name(),
            "welded_to": weld.get_path_name() if weld is not None else None,
            "world": "simulation", "force_applied": True,
            "application": application, "mass_kg": mass, "expected_delta_v": expected,
            "physics_step_limit_s": limit, "sim_time": now,
            "ends_at_sim_time": now + args["duration"] if application == "continuous" else None,
            "paused": paused, "warning": " ".join(warnings) or None}

print(json.dumps(refusable(apply)))
"""
)

# Expects ``args`` = {"paths", "tag" (or None), "world" ("auto", "simulation"
# or "editor"), "offset", "max_results", "budget"}. Reads each actor's body: in
# the running simulation (its copy, SIMULATION_WORLD) or in the editor world.
# The actors at paths come first, then the actors with the tag, each once (an
# editor path and the simulation path of the same body are one body); one page
# of them, from offset, is rendered (SCENE_READ_HELPERS.page). A body's state is
# its actor's root primitive component, else its only one. A multi-part actor
# (several primitive components, none the root) has no one body: each part that
# simulates is a body, and a part attached under it that does not simulate is
# welded into that body (physics_mode "welded", welded_to), so its entry's body
# fields are null and components holds each part's state (the first
# COMPONENTS_LISTED, with components_total), and note says why; an actor without a primitive component reports its
# transform alone, with a note. A path with no live actor, or no live copy in
# the world read, is listed in not_found, and warning says why that can be.
PHYSICS_STATE = PHYSICS_HELPERS + SCENE_READ_HELPERS + """
import math

def vector(value):
    return [value.x, value.y, value.z]

def body_of(actor):
    root = actor.root_component
    if isinstance(root, unreal.PrimitiveComponent):
        return root
    components = body_components(actor)
    return components[0] if len(components) == 1 else None

def body_state(component):
    linear = component.get_physics_linear_velocity()
    report = physics_report(component, parts=False)
    return {"component_path": component.get_path_name(), "physics_mode": report["physics_mode"],
            "linear_velocity": vector(linear),
            "angular_velocity": vector(component.get_physics_angular_velocity_in_degrees()),
            "speed": math.sqrt(linear.x * linear.x + linear.y * linear.y + linear.z * linear.z),
            "awake": bool(component.is_any_rigid_body_awake()) if report["physics_mode"] == "simulated" else None,
            "mass_kg": report["mass_kg"], "center_of_mass": vector(component.get_center_of_mass()),
            "gravity_enabled": report["gravity_enabled"], "simulate_physics": report["simulate_physics"],
            "welded_to": report["welded_to"]}

def part_state(component):
    rotation = component.get_world_rotation()
    return dict(body_state(component), component_name=component.get_name(),
                location=vector(component.get_world_location()),
                rotation=[rotation.pitch, rotation.yaw, rotation.roll])

def state(actor, simulation):
    location, rotation = actor.get_actor_location(), actor.get_actor_rotation()
    editor = editor_counterpart(actor) if simulation else actor
    entry = {"name": actor.get_actor_label(), "class_name": actor.get_class().get_name(),
             "editor_path": editor.get_path_name() if editor is not None else None,
             "simulation_path": actor.get_path_name() if simulation else None,
             "location": vector(location), "rotation": [rotation.pitch, rotation.yaw, rotation.roll]}
    component = body_of(actor)
    if component is not None:
        entry.update(body_state(component))
        return entry
    parts = body_components(actor)
    entry.update(component_path=None, physics_mode=None)
    if not parts:
        entry["note"] = "The actor has no primitive component, so it has no physics body; only its transform is read."
        return entry
    entry.update(components=[part_state(part) for part in parts[:COMPONENTS_LISTED]], components_total=len(parts),
                 note="The actor has " + str(len(parts)) + " primitive components and none is its root, so it has no "
                      "one body: each part that simulates is a body, and the parts attached under it that do not "
                      "simulate are welded into its body (physics_mode welded, welded_to names it) and move with it "
                      "as one, sharing its velocity and mass. Each part's state is in components, and the physics "
                      "tools take component_name to act on one.")
    return entry

def read():
    world = simulation_world()
    if args["world"] == "simulation" and world is None:
        return simulation_refusal("get_unreal_physics_state with world 'simulation'")
    simulation = world is not None and args["world"] != "editor"
    actors, missing, seen = [], [], set()

    def add(actor):
        key = actor.get_path_name()
        if key not in seen:
            seen.add(key)
            actors.append(actor)

    for path in args["paths"]:
        try:
            found = actor_at(path, play_world=True)
        except SimulNotFound:
            missing.append(path)
            continue
        target = simulation_counterpart(found) if simulation else editor_counterpart(found)
        if target is None:
            missing.append(path)
        else:
            add(target)
    if args["tag"]:
        pool = (unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor) if simulation
                else level_actors())
        for actor in pool:
            if live(actor) and actor.actor_has_tag(args["tag"]):
                add(actor)
    entries, next_offset, ended = page(actors, args["offset"], args["max_results"], args["budget"],
                                       lambda actor: state(actor, simulation))
    warning = None
    if missing and simulation:
        warning = ("A path in not_found names no live actor, or one with no copy in the running simulation: "
                   + _NO_COPY_REASONS + ".")
    return {"world": "simulation" if simulation else "editor", "pie_active": pie_active(),
            "sim_time": unreal.GameplayStatics.get_time_seconds(world) if simulation else None,
            "paused": bool(unreal.GameplayStatics.is_game_paused(world)) if simulation else None,
            "bodies": entries, "count": len(entries), "total": len(actors), "offset": args["offset"],
            "next_offset": next_offset, "page_ended_by": ended, "not_found": missing, "warning": warning}

print(json.dumps(read()))
"""

# Expects ``args`` = {"class_filter", "tag_filter", "simulating_only",
# "offset", "max_results", "budget"}. Lists the running simulation's live actors
# with their editor counterparts (null for one the simulation spawned, such as
# its player controller and pawn), one page from offset
# (SCENE_READ_HELPERS.page); total counts every match.
SIMULATION_ACTORS = PHYSICS_HELPERS + SCENE_READ_HELPERS + """
def simulating(actor):
    return any(physics_mode(component) == "simulated"
               for component in actor.get_components_by_class(unreal.PrimitiveComponent))

def entry(match):
    actor, moving = match
    editor = editor_counterpart(actor)
    return {"name": actor.get_actor_label(), "class_name": actor.get_class().get_name(),
            "simulation_path": actor.get_path_name(),
            "editor_path": editor.get_path_name() if editor is not None else None, "simulating": moving}

def listing():
    world = simulation_world()
    if world is None:
        return simulation_refusal("list_unreal_simulation_actors")
    found = []
    for actor in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor):
        if not live(actor):
            continue
        cls = actor.get_class()
        if args["class_filter"] and args["class_filter"] not in (cls.get_name(), cls.get_path_name()):
            continue
        if args["tag_filter"] and not actor.actor_has_tag(args["tag_filter"]):
            continue
        moving = simulating(actor)
        if args["simulating_only"] and not moving:
            continue
        found.append((actor, moving))
    entries, next_offset, ended = page(found, args["offset"], args["max_results"], args["budget"], entry)
    return {"world": world.get_path_name(), "actors": entries, "count": len(entries), "total": len(found),
            "offset": args["offset"], "next_offset": next_offset, "page_ended_by": ended,
            "sim_time": unreal.GameplayStatics.get_time_seconds(world)}

print(json.dumps(listing()))
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
# "movable", "physics"}], "folder", "tags", "transaction"}. A "physics" piece
# (a prop made physics-ready) gets the PhysicsActor preset and simulates
# physics; it is Movable already. Spawns one StaticMeshActor per
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
                if piece.get("physics"):
                    component.set_collision_profile_name("PhysicsActor")
                    component.set_simulate_physics(True)
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

#: Name of the editor-side module that keeps the last robotics scans, so the
#: pages after the first read the scan the first took instead of the level
#: again. Versioned like FORCE_MODULE.
ROBOTICS_SCAN_MODULE = "_simul_unreal_robotics_scans_v1"

# Expects ``args`` = {"analyses", "actor_filter", "criteria", "listed_limit",
# "offset", "max_results", "max_instances", "snapshot_seconds"}. Reads every
# actor loaded in the editor world whose label or object path contains
# actor_filter, ignoring case. Only primitive components with collision enabled
# count. An actor is measured by the oriented box of its physics-collision
# components: a single component's box in its own frame (a static mesh's
# bounds, a box, sphere or capsule's extent, scaled and rotated with the
# component), several components' boxes in the actor's frame; a component of
# another kind (a landscape heightfield) counts with its world-axis bounds. An
# instanced static mesh component (ISM, HISM, foliage) draws its mesh once per
# instance, not at the component, so it is no box of its own: each instance is
# measured and classified by itself (instance_index), up to max_instances per
# component. A component with more is not classified at all and is listed in
# skipped_instance_components instead: counted whole by its world-axis bounds,
# 1,500 instanced 30 cm cubes scattered over 50 x 50 m at 0-3 m read as one
# traversable surface of 2,523 m2 among the real floors (UE 5.7.4), and a
# sample of its instances would pass for all of them. The world-axis bounding
# box this used read a 20 m x 0.5 m plank
# yawed 45 degrees as 1449 x 1449 cm (210 m2 for its 10 m2), rejected a slab
# pitched 14 degrees as too thick and one rolled 180 degrees as tilted 180
# degrees (UE 5.7.4). A surface's normal is its box's thinnest axis, turned
# upward, so tilt is the angle between it and world up, area the box's other
# two sides and thickness the thin one. Static mesh collision comes from the
# mesh asset (StaticMeshEditorSubsystem): simple shapes, convex hulls and
# whether the render triangles stand in for simple collision
# (CTF_USE_COMPLEX_AS_SIMPLE). See UnrealRuntimeSession.analyze_scene_for_robotics
# for the classification. Surfaces are listed largest first, graspable objects
# by path, and each list is cut to max_results from offset; totals counts all.
# A scan is kept in ROBOTICS_SCAN_MODULE (plain values only, never Unreal
# objects, which would hold the level past a map change): offset 0 scans, and a
# later offset with the same world, analyses and filter reads that scan while
# it is under snapshot_seconds old, so paging a large level scans it once and
# every page comes from one consistent snapshot. While PIE runs the editor
# world is read (level_actors) and pie_active says so; StaticMeshEditorSubsystem
# then answers -1 and the default complexity for every mesh (observed on UE
# 5.7.4), so the mesh collision counts are null, and so is the list of actors
# missing simple collision, which those answers decide. Everything runs inside
# functions, so the script leaves no Unreal object in the console's namespace.
ROBOTICS_SCENE_ANALYSIS = EDITOR_WORLD + f"_ROBOTICS_SCAN_MODULE = {ROBOTICS_SCAN_MODULE!r}\n" + """
import math, sys, time, types

WORLD_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

def vec(value):
    return (value.x, value.y, value.z)

def along(point, axis, distance):
    return tuple(p + a * distance for p, a in zip(point, axis))

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def quat_axes(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return ((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),
            (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),
            (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)))

class Scan:
    def __init__(self):
        self.mesh_tools = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
        self.mesh_collision, self.mesh_boxes = {}, {}
        # Instanced components with more instances than are measured one by one.
        self.skipped = []

    def collision_of(self, mesh):
        key = mesh.get_path_name()
        if key not in self.mesh_collision:
            tools = self.mesh_tools
            self.mesh_collision[key] = (tools.get_simple_collision_count(mesh), tools.get_convex_collision_count(mesh),
                                        tools.get_collision_complexity(mesh).name)
        return self.mesh_collision[key]

    def mesh_box(self, mesh):
        key = mesh.get_path_name()
        if key not in self.mesh_boxes:
            box = mesh.get_bounding_box()
            low, high = vec(box.min), vec(box.max)
            self.mesh_boxes[key] = (tuple((l + h) / 2 for l, h in zip(low, high)),
                                    tuple((h - l) / 2 for l, h in zip(low, high)))
        return self.mesh_boxes[key]

    def local_box(self, component):
        # The component's box in its own unscaled frame, as (center, half sizes), or None.
        if isinstance(component, unreal.StaticMeshComponent):
            mesh = component.get_editor_property("static_mesh")
            return None if mesh is None else self.mesh_box(mesh)
        if isinstance(component, unreal.BoxComponent):
            return (0.0, 0.0, 0.0), vec(component.get_unscaled_box_extent())
        if isinstance(component, unreal.SphereComponent):
            radius = component.get_unscaled_sphere_radius()
            return (0.0, 0.0, 0.0), (radius, radius, radius)
        if isinstance(component, unreal.CapsuleComponent):
            radius = component.get_unscaled_capsule_radius()
            return (0.0, 0.0, 0.0), (radius, radius, component.get_unscaled_capsule_half_height())
        return None

    def oriented(self, local, center, axes, scale):
        for axis, offset, factor in zip(axes, local[0], scale):
            center = along(center, axis, offset * factor)
        return center, axes, tuple(abs(half * factor) for half, factor in zip(local[1], scale))

    def world_bounds(self, component):
        origin, extent, radius = unreal.SystemLibrary.get_component_bounds(component)
        return vec(origin), WORLD_AXES, vec(extent)

    def component_box(self, component):
        # The component's oriented box in world space: (center, unit axes, half sizes).
        local = self.local_box(component)
        if local is None:
            return self.world_bounds(component)
        axes = (vec(component.get_forward_vector()), vec(component.get_right_vector()),
                vec(component.get_up_vector()))
        return self.oriented(local, vec(component.get_world_location()), axes, vec(component.get_world_scale()))

    def instance_boxes(self, component, limit):
        # [(box, instance index)] for each instance, or None when there are more than limit.
        count = component.get_instance_count()
        if count > limit:
            return None
        mesh = component.get_editor_property("static_mesh")
        if mesh is None:
            return []
        local, boxes = self.mesh_box(mesh), []
        for index in range(count):
            transform = component.get_instance_transform(index, True)
            if transform is None:
                continue
            boxes.append((self.oriented(local, vec(transform.translation), quat_axes(transform.rotation),
                                        vec(transform.scale3d)), index))
        return boxes

    def actor_box(self, actor, components):
        boxes = [self.component_box(component) for component in components]
        if len(boxes) == 1:
            return boxes[0]
        origin = vec(actor.get_actor_location())
        axes = (vec(actor.get_actor_forward_vector()), vec(actor.get_actor_right_vector()),
                vec(actor.get_actor_up_vector()))
        low, high = [math.inf] * 3, [-math.inf] * 3
        for center, box_axes, half in boxes:
            for signs in ((x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)):
                corner = center
                for axis, extent, sign in zip(box_axes, half, signs):
                    corner = along(corner, axis, sign * extent)
                relative = tuple(c - o for c, o in zip(corner, origin))
                for i, axis in enumerate(axes):
                    low[i], high[i] = min(low[i], dot(relative, axis)), max(high[i], dot(relative, axis))
        center = origin
        for axis, lo, hi in zip(axes, low, high):
            center = along(center, axis, (lo + hi) / 2)
        return center, axes, tuple((hi - lo) / 2 for lo, hi in zip(low, high))

    def candidates(self, actor, physical, limit):
        # [(box, extra entry fields, components)] the actor is classified by.
        plain = [c for c in physical if not isinstance(c, unreal.InstancedStaticMeshComponent)]
        found = [(self.actor_box(actor, plain), {}, plain)] if plain else []
        for component in physical:
            if not isinstance(component, unreal.InstancedStaticMeshComponent):
                continue
            boxes = self.instance_boxes(component, limit)
            if boxes is None:
                center, _, half = self.world_bounds(component)
                self.skipped.append(entry(actor, half, {"component_path": component.get_path_name(),
                                                        "instances": component.get_instance_count()},
                                          bounds_center=[round(value, 2) for value in center]))
            else:
                found += [(box, {"component_path": component.get_path_name(), "instance_index": index}, None)
                          for box, index in boxes]
        return found

def entry(actor, half, extra, **fields):
    result = {"name": actor.get_actor_label(), "path": actor.get_path_name(),
              "class_name": actor.get_class().get_name(),
              "size": [round(2 * value, 2) for value in half]}
    result.update(extra)
    result.update(fields)
    return result

def analyze():
    analyses = set(args["analyses"])
    criteria = args["criteria"]
    needle = (args["actor_filter"] or "").lower()
    pie = pie_active()
    scan = Scan()
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
                simple, convex, complexity = scan.collision_of(mesh)
                summary["simple_shapes"] += simple
                summary["convex_hulls"] += convex
                if complexity == "CTF_USE_COMPLEX_AS_SIMPLE":
                    summary["complex_collision"] += 1
                elif physics and simple + convex == 0:
                    summary["missing_simple_collision"] += 1
                    missing = True
        if missing and len(summary["missing_simple_collision_actors"]) < args["listed_limit"]:
            summary["missing_simple_collision_actors"].append(actor.get_path_name())
        if not physical or not ({"traversability", "graspability"} & analyses):
            continue
        root = actor.root_component
        movable = root is not None and root.mobility.name == "MOVABLE"
        for (center, axes, half), extra, components in scan.candidates(actor, physical, args["max_instances"]):
            if "traversability" in analyses:
                thin = min(range(3), key=lambda i: half[i])
                sides = [half[i] for i in range(3) if i != thin]
                normal = axes[thin] if axes[thin][2] >= 0 else tuple(-value for value in axes[thin])
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, normal[2]))))
                area = 4 * sides[0] * sides[1] / 10000.0
                if (area >= criteria["surface_min_area_m2"]
                        and half[thin] <= criteria["surface_max_thickness_ratio"] * min(sides)
                        and tilt <= criteria["surface_max_tilt_degrees"]):
                    traversable.append(entry(actor, half, extra, top_z=round(center[2] + normal[2] * half[thin], 2),
                                             area_m2=round(area, 3), tilt_degrees=round(tilt, 2),
                                             normal=[round(value, 4) for value in normal]))
            if "graspability" in analyses and movable and 0 < 2 * max(half) <= criteria["grasp_max_size_cm"]:
                bodies = components if components is not None else []
                mass = sum(component.get_mass() for component in bodies)
                simulated = any(component.get_editor_property("body_instance").get_editor_property("simulate_physics")
                                for component in (bodies or physical))
                graspable.append(entry(actor, half, extra, simulate_physics=bool(simulated),
                                       mass_kg=round(mass, 3) if mass > 0 else None))
    if pie:
        for key in ("simple_shapes", "convex_hulls", "complex_collision", "missing_simple_collision",
                    "missing_simple_collision_actors"):
            summary[key] = None
    traversable.sort(key=lambda item: (-item["area_m2"], item["path"], item.get("instance_index") or 0))
    graspable.sort(key=lambda item: (item["path"], item.get("instance_index") or 0))
    skipped = sorted(scan.skipped, key=lambda item: (item["path"], item["component_path"]))
    warning = None
    if skipped:
        warning = (str(len(skipped)) + " instanced mesh component(s) have more than " + str(args["max_instances"])
                   + " instances (" + str(sum(item["instances"] for item in skipped)) + " in all), more than are "
                   "measured one by one, so none of their instances was classified: they may hold floors or "
                   "graspable props this analysis does not list. They are in skipped_instance_components, with "
                   "their world bounds.")
    return {"traversable_surfaces": traversable, "graspable_objects": graspable,
            "collision_summary": summary if "collision_complexity" in analyses else {},
            "total_actors_analyzed": summary["total_actors"], "pie_active": pie,
            "skipped_instance_components": skipped[:20],
            "skipped_instance_components_count": len(skipped), "warning": warning}

def scans():
    module = sys.modules.get(_ROBOTICS_SCAN_MODULE)
    if module is None:
        module = types.ModuleType(_ROBOTICS_SCAN_MODULE)
        module.scans = {}
        sys.modules[_ROBOTICS_SCAN_MODULE] = module
    return module.scans

def run():
    world = editor_world()
    key = json.dumps([world.get_path_name() if world is not None else None, sorted(args["analyses"]),
                      (args["actor_filter"] or "").lower()])
    kept = scans()
    now = time.time()
    cached = kept.get(key)
    if args["offset"] == 0 or cached is None or now - cached["scanned_at"] > args["snapshot_seconds"]:
        cached = {"scanned_at": now, "result": analyze()}
        kept[key] = cached
        for stale in sorted(kept, key=lambda name: kept[name]["scanned_at"])[:-4]:
            del kept[stale]
    full = cached["result"]
    page = slice(args["offset"], args["offset"] + args["max_results"])
    return dict(full, traversable_surfaces=full["traversable_surfaces"][page],
                graspable_objects=full["graspable_objects"][page],
                totals={"traversable_surfaces": len(full["traversable_surfaces"]),
                        "graspable_objects": len(full["graspable_objects"])},
                scanned_at=cached["scanned_at"], snapshot_age_s=round(now - cached["scanned_at"], 3))

print(json.dumps(run()))
"""

# The Play In Editor state get_viewport_info adds to the level viewport camera:
# whether PIE runs, the player camera (PlayerCameraManager location, rotation
# and field of view, and the game viewport's size from WidgetLayoutLibrary)
# whenever the game world has a player controller, and the level viewport
# layout's panes. Simulate In Editor has one too (a SpectatorPawn at the player
# start, observed on UE 5.7.4), whose camera is not what the level viewport
# shows the simulation from.
VIEWPORT_STATE = "import json, unreal\n" + in_function("""
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
""")

# Expects ``args`` = {"path", "written"}: the newest file of a HighResShot at
# the viewport's own size, and every file it wrote. Reads the newest one's PNG
# header for the viewport size and deletes them all.
MEASURED_SHOT = "import json, os, struct\n" + in_function("""
with open(args["path"], "rb") as stream:
    head = stream.read(24)
for path in args["written"]:
    if os.path.isfile(path):
        os.remove(path)
if head[:8] != b"\\x89PNG\\r\\n\\x1a\\n":
    raise ValueError("Not a PNG: " + args["path"])
width, height = struct.unpack(">II", head[16:24])
print(json.dumps({"width": width, "height": height}))
""")

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
                "details": {"code": "TooManyActors", "would_change": len(targets),
                            "max_actors": args["max_actors"]}}
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
# into the editor's screenshot directory. The source's show-only and hidden
# actor lists and show flags are not copied.
#
# Outside PIE the capture is spawned in the editor world through
# EditorActorSubsystem as a transient actor, which no listing shows, and
# destroyed afterwards, all inside one transaction: spawning and destroying are
# each an undoable editor transaction, and on their own one Undo brought the
# capture actor back and the undo buffer kept its component and render target
# alive until the map changed (UE 5.7.4: five 3840x2160 captures held 158 MB).
# When no transaction was open (begin_transaction returned 0) it is cancelled,
# leaving nothing in the undo history. When one was (an edit the user is
# making, such as a drag in the viewport or on a Details slider, which keeps a
# transaction open across frames), cancelling ours would cancel theirs with it
# (UE 5.7.4 logs "Canceling transaction partially is unsupported" and drops the
# whole edit), so ours is ended instead and the spawn and removal fold into
# theirs, where undoing it undoes the user's edit and leaves no capture actor.
# The render target is detached and released once the image is written, in
# every world, and each step of the clean-up runs even if the one before it
# raised.
#
# While PIE runs, a path of the running game's actor (as PIE names it) is
# captured as given; an editor actor's counterpart in the game world is the
# game actor with its object name, of its class, which also finds an actor
# World Partition streamed into a cell level of its own
# (/Memory/UEDPIE_0_<map>_<cell>.<map>:PersistentLevel.<name>), and with its
# actor GUID when both have one (PIE copies keep it, observed on UE 5.7.4), so
# a game actor that merely took the name is not taken; a name that is not
# unique and that no GUID settles finds nothing. The capture is spawned in the game
# world (GameplayStatics' deferred spawn, which Unreal's Python reaches only
# through call_method) and destroyed; an editor actor without a counterpart
# (not streamed in, or destroyed by the game) is refused, as the editor
# spawns nothing during PIE. The image is returned as base64 when it is at
# most ``inline_limit`` bytes (0: never). Everything runs inside capture(), so
# no world or render target stays bound in the editor's Python namespace.
CAMERA_CAPTURE = ACTOR_HELPERS + """
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

def guid_of(actor):
    try:
        text = actor.get_editor_property("actor_guid").to_string()
    except Exception:
        return None
    return text if text.strip("0") else None

def counterpart(actor, game):
    name, guid = actor.get_name(), guid_of(actor)
    found = [candidate for candidate in unreal.GameplayStatics.get_all_actors_of_class(game, actor.get_class())
             if candidate.get_name() == name]
    if guid is not None:
        found = [candidate for candidate in found if guid_of(candidate) in (guid, None)]
    return found[0] if len(found) == 1 else None

def no_camera(source):
    return {"success": False, "error_type": "ValidationError",
            "error": source.get_path_name() + " (a " + source.get_class().get_name() + ") has no camera or "
                     "scene capture component: pass a CameraActor, CineCameraActor, SceneCapture2D or an actor "
                     "with a CameraComponent or SceneCaptureComponent2D",
            "details": {"code": "NotACamera", "actor_class": source.get_class().get_name()}}

def write_image(capture_component, target_texture, label):
    capture_component.capture_scene()
    directory = unreal.Paths.convert_relative_path_to_full(unreal.Paths.screen_shot_dir())
    os.makedirs(directory, exist_ok=True)
    extension = "jpg" if args["format"] == "jpeg" else "png"
    path = os.path.join(directory, "SimulCamera_%s_%d.%s" % (label, int(time.time() * 1000), extension))
    options = unreal.ImageWriteOptions()
    options.format = unreal.DesiredImageFormat.JPG if extension == "jpg" else unreal.DesiredImageFormat.PNG
    options.compression_quality = args["quality"]
    options.async_ = False
    options.overwrite_file = True
    unreal.ImageWriteBlueprintLibrary.export_to_disk(target_texture, path, options)
    return path

def configure(capture_component, lens, target, target_texture):
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

def release(capture_component, target_texture):
    if capture_component is not None:
        capture_component.set_editor_property("texture_target", None)
    if target_texture is not None:
        unreal.RenderingLibrary.release_render_target2d(target_texture)

def close_transaction(index):
    if index == 0:
        unreal.SystemLibrary.cancel_transaction(index)
    else:
        unreal.SystemLibrary.end_transaction()

def capture_in_editor(world, target, lens, location, rotation, label):
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    transaction = unreal.SystemLibrary.begin_transaction("simul", "simul: capture camera", None)
    spawned = capture_component = target_texture = None
    try:
        spawned = actors.spawn_actor_from_class(unreal.SceneCapture2D, location, rotation, True)
        if spawned is None:
            return None
        capture_component = spawned.capture_component2d
        target_texture = unreal.RenderingLibrary.create_render_target2d(
            world, args["width"], args["height"], unreal.TextureRenderTargetFormat.RTF_RGBA8,
            unreal.LinearColor(0.0, 0.0, 0.0, 1.0), False)
        configure(capture_component, lens, target, target_texture)
        return write_image(capture_component, target_texture, label)
    finally:
        try:
            release(capture_component, target_texture)
        finally:
            try:
                if spawned is not None:
                    actors.destroy_actor(spawned)
            finally:
                close_transaction(transaction)

def capture_in_game(world, target, lens, location, rotation, label):
    transform = unreal.Transform(location=location, rotation=rotation, scale=unreal.Vector(1.0, 1.0, 1.0))
    statics = unreal.get_default_object(unreal.GameplayStatics)
    spawned = statics.call_method("BeginDeferredActorSpawnFromClass",
                                  (world, unreal.SceneCapture2D.static_class(), transform))
    if spawned is None:
        return None
    capture_component = target_texture = None
    try:
        capture_component = spawned.capture_component2d
        target_texture = unreal.RenderingLibrary.create_render_target2d(
            world, args["width"], args["height"], unreal.TextureRenderTargetFormat.RTF_RGBA8,
            unreal.LinearColor(0.0, 0.0, 0.0, 1.0), False)
        configure(capture_component, lens, target, target_texture)
        statics.call_method("FinishSpawningActor", (spawned, transform))
        return write_image(capture_component, target_texture, label)
    finally:
        try:
            release(capture_component, target_texture)
        finally:
            spawned.destroy_actor()

def capture():
    pie = pie_active()
    game = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world() if pie else None
    source = actor_at(args["actor_path"], play_world=game is not None)
    if lens_of(source) is None:
        return no_camera(source)
    target, world, world_kind = source, editor_world(), "editor"
    if game is not None:
        world, world_kind = game, "game"
        if source.get_world() != game:
            target = counterpart(source, game)
            if target is None:
                refusal = pie_refusal("capture an actor that is not in the running game")
                refusal["error"] += (": " + source.get_path_name() + " has no counterpart there (World "
                                     "Partition has not streamed it in, or the game destroyed it). Pass a "
                                     "camera of the running game by its path, or stop PIE")
                return refusal
    lens = lens_of(target)
    if lens is None:
        return no_camera(target)
    component = lens["component"]
    location, rotation = component.get_world_location(), component.get_world_rotation()
    label = re.sub("[^A-Za-z0-9_-]+", "_", target.get_actor_label())[:48] or "Camera"
    if world_kind == "editor":
        path = capture_in_editor(world, target, lens, location, rotation, label)
    else:
        path = capture_in_game(world, target, lens, location, rotation, label)
    if path is None:
        return {"success": False, "error_type": "EngineError", "error": "Unreal spawned no scene capture"}
    size = os.path.getsize(path) if os.path.isfile(path) else 0
    if not size:
        return {"success": False, "error_type": "EngineError", "error": "Unreal wrote no image to " + path}
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
