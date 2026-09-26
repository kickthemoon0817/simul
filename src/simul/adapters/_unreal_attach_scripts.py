"""Fixed editor identity and named-control scripts, using supported UE APIs."""

# Keep callbacks in their own module: Remote Control's public Python globals are
# shared with arbitrary user scripts. Do not keep references to UWorld objects.
STATE_MODULE = """
import os, uuid, unreal
instance_id = uuid.uuid4().hex
document_id = uuid.uuid4().hex

def map_changed(flags):
    global document_id
    # MapChangeEventFlags::NewMap | WorldTornDown; rebuilds do not detach.
    if flags & 5:
        document_id = uuid.uuid4().hex

unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).on_map_changed.add_callable(map_changed)

def identity():
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    if world is None:
        raise RuntimeError("No editor world is available")
    level = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    return {
        "instance_id": instance_id, "document_id": document_id, "pid": os.getpid(),
        "project_path": os.path.abspath(unreal.Paths.get_project_file_path()),
        "map_path": world.get_path_name(),
        "engine_version": unreal.SystemLibrary.get_engine_version(),
        "viewports": [str(k) for k in level.get_viewport_config_keys()],
        "active_viewport": str(level.get_active_viewport_config_key()),
        "dirty_map_packages": [p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages()],
    }

def verify(expected):
    actual = identity()
    for key in ("instance_id", "document_id", "project_path", "map_path"):
        if actual[key] != expected[key]:
            raise RuntimeError("Unreal attachment changed (" + key + "); run simul unreal attach again")
    if expected["viewport"] not in actual["viewports"]:
        raise RuntimeError("Attached viewport is unavailable; run simul unreal attach again")
"""

EDITOR_STATE = (
    "import sys, types, json, unreal\n"
    "if '_simul_unreal_attachment_v1' not in sys.modules:\n"
    "    _simul_new = types.ModuleType('_simul_unreal_attachment_v1')\n"
    f"    exec({STATE_MODULE!r}, _simul_new.__dict__)\n"
    "    sys.modules['_simul_unreal_attachment_v1'] = _simul_new\n"
    "_simul_editor = sys.modules['_simul_unreal_attachment_v1']\n"
)

CONTROL_UI = """
import json, unreal
level = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
action = args["action"]
viewport = args["viewport"]
if viewport not in [str(k) for k in level.get_viewport_config_keys()]:
    raise ValueError("Attached viewport is unavailable; attach again")
overlay = getattr(unreal, "SimulAgentOverlayLibrary", None)
result = {"agent_control": action, "execution_method": "unreal_editor_api", "viewport": viewport,
          "agent_id": args["agent_id"], "system_cursor_moved": False, "overlay_available": overlay is not None}
if action in ("move_cursor", "clear_cursor") and overlay is None:
    raise RuntimeError("Agent overlay unavailable. Close this project and run "
                       "simul unreal setup <project> --agent-overlay --yes, then attach again")
if action == "inspect":
    result.update(_simul_editor.identity())
    result["selected_actors"] = [{"path": a.get_path_name(), "label": a.get_actor_label()}
                                  for a in actors.get_selected_level_actors()]
    result["is_playing"] = level.is_in_play_in_editor()
    result["game_view"] = level.editor_get_game_view(viewport)
    pilot = level.get_pilot_level_actor(viewport)
    result["pilot_actor"] = pilot.get_path_name() if pilot else None
    result["actions"] = args["actions"]
    result["agent_cursors"] = json.loads(overlay.inspect_cursors(viewport))["agent_cursors"] if overlay else []
elif action == "clear_cursor":
    result["removed"] = overlay.clear_cursor(viewport, args["agent_id"])
else:
    if level.is_in_play_in_editor():
        raise ValueError("Stop PIE before changing editor controls")
    actor = None
    if action in ("select_actor", "set_property", "pilot_actor"):
        target = args["target"]
        candidates = ([a for a in actors.get_all_level_actors()
                       if target in (a.get_path_name(), a.get_actor_label())]
                      if target else list(actors.get_selected_level_actors()))
        if len(candidates) != 1:
            raise ValueError("Select exactly one actor or supply its unique full path")
        actor = candidates[0]
        result["actor_path"] = actor.get_path_name()
    if action == "select_actor":
        actors.set_selected_level_actors([actor])
        if list(actors.get_selected_level_actors()) != [actor]:
            raise RuntimeError("Unreal did not select the requested actor")
    elif action == "clear_selection":
        actors.set_selected_level_actors([])
    elif action == "set_property":
        key, values = args["property_name"], args["value"]
        with unreal.ScopedEditorTransaction("Simul: set actor " + key):
            actor.modify()
            if key == "location":
                actor.set_actor_location(unreal.Vector(*values), False, True)
                v = actor.get_actor_location()
                result["value"] = [v.x, v.y, v.z]
            elif key == "rotation":
                actor.set_actor_rotation(unreal.Rotator(pitch=values[0], yaw=values[1], roll=values[2]), True)
                v = actor.get_actor_rotation()
                result["value"] = [v.pitch, v.yaw, v.roll]
            elif key == "scale":
                actor.set_actor_scale3d(unreal.Vector(*values))
                v = actor.get_actor_scale3d()
                result["value"] = [v.x, v.y, v.z]
        result["property_name"] = key
        if key != "rotation" and any(abs(a - b) > 0.001 for a, b in zip(result["value"], values)):
            raise RuntimeError("Actor transform readback differs from the requested value")
    elif action == "pilot_actor":
        level.pilot_level_actor(actor, viewport)
        if level.get_pilot_level_actor(viewport) != actor:
            raise RuntimeError("Unreal did not pilot the requested actor")
    elif action == "eject_actor":
        level.eject_pilot_level_actor(viewport)
        if level.get_pilot_level_actor(viewport) is not None:
            raise RuntimeError("Unreal did not eject the piloted actor")
    elif action == "set_game_view":
        level.editor_set_game_view(args["enabled"], viewport)
        result["enabled"] = level.editor_get_game_view(viewport)
        if result["enabled"] != args["enabled"]:
            raise RuntimeError("Unreal did not change game view")
    level.editor_invalidate_viewports()
    if overlay:
        # Only explicit move_cursor changes a marker's position. Subsequent actions
        # keep the agent's pointer where it was placed and update its activity.
        previous = json.loads(overlay.inspect_cursors(viewport))["agent_cursors"]
        point = next((m["position"] for m in previous if m["agent_id"] == args["agent_id"]), [0.5, 0.5])
        if action == "move_cursor":
            point = args["position"] or [0.5, 0.5]
            label = args["activity"] or "pointing"
        else:
            detail = args["property_name"] or args["target"] or ""
            label = (action + (": " + detail if detail else "") + " (done)")[:256]
        cursor = json.loads(overlay.update_cursor(viewport, args["agent_id"], unreal.Vector2D(*point), label))
        if cursor.get("error"):
            # The editor action has already completed; don't report it as failed
            # and encourage a duplicate mutation because its annotation failed.
            if action == "move_cursor":
                raise RuntimeError(cursor["error"])
            result["overlay_error"] = cursor["error"]
        else:
            result["agent_cursor"] = cursor
print(json.dumps(result))
"""
