# Attach Simul to an existing Blender window

Simul has two Blender connection modes:

- `embedded` (default): `bpy` runs in the MCP server's Python process. This
  does not connect to an already open Blender application.
- `attached`: tools execute in the exact Blender process and window selected
  by `simul blender attach`. The MCP server does not need `bpy` installed.

Attachment does not launch Blender, reload a file, save the scene, discard
unsaved work, or fall back to an embedded scene when the editor is unavailable.

## Enable the bridge once in the running application

```sh
simul blender install-bridge
```

This builds `~/.simul/blender/simul_blender_bridge.zip`. In the **existing
Blender window**, open Preferences → Add-ons → Install from Disk, select the
ZIP, and enable **Simul Blender Bridge**. No restart or scene reload is
needed. The bridge must be enabled inside the process to control its memory;
installing `bpy` in an external Python environment cannot provide that access.

The ZIP contains the shared Blender operations and filesystem policy plus a
standard-library transport. It does not install FastMCP, Pydantic, or other
server dependencies into Blender. After upgrading Simul, rebuild and reinstall
the ZIP. Restart Blender before attaching again so all loaded bridge modules
use the new version; Simul does not restart the editor automatically.

## Select and verify the target

```sh
simul blender instances
simul blender attach --instance <instance-id> --window <window-id>
simul blender status
simul server --backends blender --blender-mode attached
```

`instances` reports the PID, Blender version, file path (null for an unsaved
file), dirty state, and every window's ID, scene, and workspace. A process
without the bridge is not discoverable by these commands.

When exactly one live GUI process and one window exist, `simul blender attach`
can omit both selectors. Ambiguous selections fail and require explicit IDs.
Window IDs are session identities, not OS window titles or coordinates.

The existing Blender and SimReady MCP tools use this connection. For example,
`get_blender_info` reports the selected process/window and `create_blender_object`
creates an object in that window's scene. The bridge supplies its 3D View context
when available, without moving the OS pointer or changing application focus.
Granular object lookups are limited to the selected scene; Blender datablocks
that are linked between scenes remain shared, just as in Blender itself.

For an MCP client configuration, add `--blender-mode attached` to the server
arguments, or set `BLENDER__MODE=attached` in the server environment. Restart
the MCP server if it was already running in embedded mode. Subsequent explicit
`attach`/`detach` commands take effect on its next tool call without restarting it.

## Named agent controls

Call `control_blender_ui` with an `agent_control` action. A single structured
request replaces generated Python and hardcoded display coordinates for the
supported actions:

```json
{"agent_control": "inspect"}
{"agent_control": "open_menu", "target": "add"}
{"agent_control": "set_tool", "target": "move"}
{"agent_control": "select_object", "target": "Cube"}
{"agent_control": "show_properties", "target": "OBJECT"}
{"agent_control": "set_property", "target": "location", "value": [1, 2, 3]}
{"agent_control": "move_cursor", "position": [0.3, 0.7], "agent_id": "builder"}
{"agent_control": "move_cursor", "position": [0.8, 0.6], "agent_id": "reviewer"}
{"agent_control": "clear_cursor", "agent_id": "reviewer"}
```

Each line is a separate tool call. `inspect` reports live editor IDs, mode,
active object, active tool, Properties tabs, workspace sharing, visible agent
marker records, and supported targets. For editor actions with multiple matching
3D Views or Properties editors, supply `area_id`; the tool rejects an
ambiguous or stale editor instead of guessing. UI controls require attached
mode and inherit its process, document, window and scene checks. Object selection
and property writes do not need a 3D View or `area_id`; when available, their
annotation appears in the largest viewport, or the explicitly selected viewport.

| Action | Target / behavior |
|---|---|
| `inspect` | Editors, active object/tool, Properties tabs, workspace sharing and agent markers |
| `open_menu` | `add`, `object`, `view`; Object Mode only; reports menu requested |
| `set_tool` | `select_box`, `move`, `rotate`, `scale`; refuses shared workspaces and verifies active tool |
| `isolate_workspace` | Explicitly duplicate the attached window's shared workspace; call `inspect` again for new editor IDs |
| `select_object` | Visible, selectable object name in the current view layer; replaces selection in Object Mode |
| `show_properties` | Blender tab identifier such as `OBJECT`, `RENDER`, `MODIFIER`; unavailable tabs return an error |
| `set_property` | Active editable object's `location`, `rotation_euler` or `scale`; three finite values, radians for rotation, Object Mode only |
| `move_cursor` | Draw the agent's virtual pointer; center by default, optional normalized `position: [x, y]` measured from bottom-left |
| `clear_cursor` | Remove only `agent_id`'s marker in the attached window |

### Agent cursors and action visibility

`move_cursor` now draws a **virtual agent pointer** inside Blender. It never
calls `Window.cursor_warp`, injects OS mouse events, or changes Blender's 3D
cursor. It returns `cursor_kind: "agent_overlay"` and `system_cursor_moved: false`.
The pointer is an annotation, not an input device or a button hit-test target.

Each `agent_id` has its own colored pointer and label, plus a latest-action
status in the editor. Named actions update that status; subsequent actions in
the same editor preserve the pointer's chosen position. Labels describe
completed actions or requested menus, not a guarantee that an agent is still
working. Inspection/isolation/clear operations do not create a new marker.

Use stable distinct labels such as `builder` and `reviewer`. Omitting `agent_id`
uses the MCP session identity. Labels are display identifiers, not authentication
or locks; Blender operations still execute sequentially. Each agent has one
latest marker per window. Markers expire after 120 seconds, are capped at 64
across the process, and disappear when their scene/workspace/editor becomes
stale, another file loads, or the bridge stops. They are transient overlays,
not saved scene objects. Popups may cover them, as with other editor overlays.

The implementation uses Blender's
[`SpaceView3D.draw_handler_add`](https://docs.blender.org/api/4.2/bpy.types.SpaceView3D.html#bpy.types.SpaceView3D.draw_handler_add)
and Properties draw handlers. These annotations are independent of native menu
placement: menus still open through Blender's UI API using Blender's event
context, not at the virtual pointer as if it were a physical mouse.

### Workspace tool isolation

Blender's active tool belongs to a workspace and mode, so an editor context
alone cannot confine it to one window. `inspect.shared_window_ids` reports
other windows using the selected workspace. `set_tool` refuses to change a
shared workspace. To work independently, call:

```json
{"agent_control": "isolate_workspace"}
{"agent_control": "inspect"}
```

Isolation explicitly creates a workspace copy for the target window; it does
not copy scene objects or save the file. Blender switches workspaces after the
request returns, so `workspace_copy_requested` requires a fresh inspection and
new editor IDs before further editor actions. Other windows retain their
workspace/tool state. Repeating isolation after the switch is a no-op. Linked
child windows follow their parent's workspace and cannot be isolated this way;
use an independent main Blender window. Within a single workspace, all views
of the same mode still share the active tool. Shared scene/object datablocks
also retain Blender's usual sharing semantics.

These actions use Blender's UI/data APIs. Menu and toolbar targets invoke their
named operations; the virtual marker indicates the action's editor, not an
estimated button coordinate.
The response identifies `execution_method: "blender_ui_api"`. The tool does
not find arbitrary buttons, inject clicks/keystrokes, provide strict
mouse/keyboard-only replay, or dismiss modal dialogs. Menu completion means
the request was accepted, not that a menu item was selected. A modal dialog
or busy editor may prevent progress; inspect the UI before retrying.

The bounded actions work with `security.allow_script_execution=false`; no
arbitrary Python, operator names, or property paths are accepted. MCP client
approval settings still apply. There is no general OS input-injection backend
or bypass of operating-system permissions in this implementation.

## Target changes and timeouts

Every request verifies the bridge instance, file-load generation, window and
scene identity before executing. Closing the window, switching its scene,
loading another `.blend`, restarting Blender, or restarting the bridge makes
the old target invalid. Run `instances` and `attach` again. Simul never silently
selects a replacement window. An explicit `open_blender_file` affects the whole
Blender process and invalidates its old attachment after completing.

Object-creation and other mode-sensitive granular operations refuse to run
in Edit/Pose Mode rather than unexpectedly editing the active mesh. Scripts
remain responsible for their own operator context and mode changes.

Blender's application timer polls a nonblocking loopback socket and runs all
`bpy` work on the main thread. Requests execute sequentially. A long render or
script can make the GUI temporarily unresponsive. The client timeout limits
waiting, not execution: an operation already started cannot safely be stopped.
Its outcome is unknown after a timeout, and Simul never automatically retries it.
Requests whose deadlines expired before execution are refused.

The transport uses a per-instance token in owner-only discovery/attachment
files. It accepts loopback connections only. Normal filesystem sandbox settings
still apply, including to saving an existing file in place. Arbitrary scripting
is controlled by `security.allow_script_execution` as for embedded mode.

```sh
simul blender detach
```

Detaching forgets the selection; Blender stays open with its current scene.
Disable the add-on in Blender to stop its listener.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `BLENDER__MODE` | `embedded` | Choose local bpy or an attached editor |
| `BLENDER__DISCOVERY_DIR` | `~/.simul/blender` | Directory searched by the CLI |
| `BLENDER__ATTACHMENT_PATH` | `~/.simul/blender/attachment.json` | Selection used by this client/server |
| `BLENDER__CONNECTION_TIMEOUT` | `30` | Maximum seconds to wait per operation |
| `SIMUL_BLENDER_DISCOVERY_DIR` | `~/.simul/blender` | Override discovery location in Blender's environment |

Use distinct attachment paths for independent MCP servers controlling different
windows. Set the same path on each server and the CLI used to select its target.
For a custom discovery directory, set the matching directory in both environments.

## Verification

The GUI integration test runs in a new disposable process and never uses an
existing user's editor. It checks preservation of unsaved work, main-thread
execution, named UI actions with scripting disabled, object editing, JPEG capture, multiple windows, scene/file changes,
and detachment. It always closes the process it started.

```sh
pytest tests/blender --no-cov
SIMUL_BLENDER_LIVE=1 pytest tests/blender/test_live_attach.py -m blender_live --no-cov
```

The bridge targets Blender 4.2+ APIs; live verification was performed on Blender
5.0.1 on macOS. It does not add a headless job launcher or fix every pre-existing
Blender tool limitation.
