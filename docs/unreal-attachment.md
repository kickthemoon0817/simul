# Attach Simul to an existing Unreal editor

Unreal now supports the explicit attachment workflow used by the Blender backend.
The existing Remote Control connection remains the default (`endpoint` mode).
`attached` mode reads a saved selection and verifies the editor process, project,
map generation and level viewport before dispatch. It does not start the editor,
load or save a map, or select a replacement editor when the target disappears.

## Discover and select

The existing editor needs RemoteControl and PythonScriptPlugin enabled, with
Remote Control Python execution allowed. Use `simul unreal setup` on your project
for initial configuration; attachment itself does not modify project files.
If setup changes configuration for an already running editor, restart it after
saving your work. No additional Unreal plugin is required by attachment.

```sh
simul unreal instances
simul unreal instances --host localhost --port 30010
simul unreal attach --host localhost --port 30010 --viewport '<key from instances>'
simul unreal status
simul server --backends unreal --unreal-mode attached
```

Discovery reports process identity, project path, loaded map, dirty map packages,
and level viewport config keys. With one editor and one viewport, selectors may
be omitted. Unreal layouts often contain several viewports: supply a key explicitly
when selection is ambiguous. These keys identify level viewports, not operating
system windows or arbitrary asset-editor tabs.

The selection is stored in `~/.simul/unreal/attachment.json` with owner-only
permissions on POSIX. Passphrases are not copied into that file; configure
`UNREAL__PASSPHRASE` on the client when the editor requires authentication.
Discovery uses `UNREAL__HOST`, `UNREAL__PORT`, and the existing scan-port range.

## Named editor controls

`control_unreal_ui` is available in both thin and full MCP surfaces. It works
with `SECURITY__ALLOW_SCRIPT_EXECUTION=false`, using a bounded list of actions:

| Action | Inputs and result |
|---|---|
| `inspect` | Editor/map identity, dirty maps, selected actors, pilot actor, game view and available actions |
| `select_actor` | `target`: full actor path or unique label in the current editor world |
| `clear_selection` | Clear editor actor selection |
| `set_property` | `property_name`: `location`, `rotation` or `scale`; `value`: three finite numbers |
| `pilot_actor` | Pilot `target` in the attached level viewport |
| `eject_actor` | Stop piloting in that viewport |
| `set_game_view` | `enabled`: boolean; returns the state read from the viewport |

For actor actions, omit `target` to use the sole selected actor. Ambiguous labels,
missing actors and stale attachments fail explicitly. Location uses centimeters;
rotation uses Pitch/Yaw/Roll in degrees and returns Unreal's normalized angles.
Transform editing participates in an editor undo transaction. Stop PIE before
changing editor controls; inspection is available while PIE is running.

CLI equivalents:

```sh
simul unreal control inspect
simul unreal control select_actor --target 'Cube'
simul unreal control set_property --property location --value '[100, 0, 200]'
simul unreal control pilot_actor --target 'CameraActor'
simul unreal control eject_actor
simul unreal control set_game_view --enabled
```

`status` and `control` always use the saved attachment. To make other existing CLI
operations use it as well, set `UNREAL__MODE=attached`. Existing MCP tools also use
the selected endpoint when the server runs with `--unreal-mode attached`.
The default `endpoint` mode continues to use host/port configuration.

## Identity checks and limits

The first discovery call installs a small in-memory Python module and a map-change
listener. It changes no scene data. Opening/replacing a map, restarting the editor,
or removing the selected viewport invalidates the selection. Re-run `instances`
and `attach`; a changed target is never accepted automatically. Reattaching or
detaching takes effect on the next tool session without restarting the MCP server.

Named controls and Python-backed tools check identity inside the same editor Python
execution as their operation. Legacy Remote Control calls check immediately before
dispatch in a separate request: attachment is not an editor lock, so avoid replacing
maps concurrently with those calls. Viewport config keys can be reused by Unreal
layouts; reattach after reorganizing level-editor windows. An explicit arbitrary
script can still change maps or perform other actions allowed by its execution policy.

Named controls use Unreal's editor APIs. They do not inject OS mouse/keyboard input,
open arbitrary menus, address asset-editor tabs, or dismiss modal dialogs. A busy
editor can time out. Mutations are not automatically retried; a timeout after dispatch
means the operation may have executed, so inspect before retrying.

```sh
simul unreal detach
```

Detaching only forgets the selection. The editor and unsaved work remain open.
Use distinct `UNREAL__ATTACHMENT_PATH` values for clients targeting different editors.

## Verification

```sh
pytest tests/unreal/test_attachment.py --no-cov
# Only against a copied, disposable project: this test replaces the scratch map.
SIMUL_UNREAL_ATTACH_LIVE=1 UNREAL__PORT=30019 \
  pytest tests/unreal/test_live_attachment.py -m unreal_live --no-cov
```

The live test exercises the named MCP actions with arbitrary scripting disabled,
checks dirty-map preservation on attach, rejects replaced-map targets and verifies
reattachment. Implemented against Unreal 5.7 editor APIs; other versions require
verification of their Python exposure before claiming compatibility.
