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

`simul unreal instances` and the `list_unreal_instances` MCP tool run the same
discovery and return the same payload; `--no-scan` is the tool's `scan: false`.
Discovery probes the scan range (`UNREAL__SCAN_PORT_START` up to, not including,
`UNREAL__SCAN_PORT_END`; 30010–30019 by default) plus the configured and attached
ports on `UNREAL__HOST`, or on the attachment's host in attached mode. `--port`
probes only that port, `--host` another host, and `--no-scan` only the active
endpoint. Probes run concurrently, at most 32 at a time. Each step of a probe
is bounded by `UNREAL__PING_TIMEOUT`, and every request carries
`UNREAL__PASSPHRASE` when it is set. Each instance reports:

| Field | Meaning |
|---|---|
| `name` | `default` for the active endpoint, otherwise `unreal-<port>` |
| `host`, `port` | Remote Control endpoint |
| `reachable`, `latency_ms` | Whether Remote Control answered `GET /remote/info` (an HTTP 401 refusal counts), and its round trip |
| `active` | Unreal tools dispatch here: the configured endpoint, or the attached editor in attached mode |
| `attached` | Editor, loaded map and viewport match the saved attachment (reported in either mode) |
| `error` | Why the endpoint is unreachable, refused the request or has no identity available; otherwise null |
| `instance_id`, `document_id`, `pid` | Editor process identity, loaded-map generation and process ID |
| `engine_version`, `project_name`, `project_path`, `map_path` | Engine, project and loaded map |
| `viewports`, `active_viewport` | Level viewport config keys, and the active one |
| `dirty_map_packages` | Map packages with unsaved changes |

The identity fields come from the editor's Python. They are null (the lists
empty) when it does not answer, for example with Python execution disabled, an
editor still starting up or one too busy to answer within the timeout; `error`
then gives the reason, the editor is still `reachable`, and it cannot be
attached until it answers.
An editor whose Remote Control refuses the request with HTTP 401 (it enforces a
passphrase and `UNREAL__PASSPHRASE` is missing or wrong) is reported the same
way, with `error` saying to set `UNREAL__PASSPHRASE`. `ping_unreal` and
`unreal_health_check` apply the same rule: Remote Control answered, so
`reachable` is true, and the call fails with `error_type: AuthenticationError`
and that `error` (`unreal_health_check`'s deprecated `connected` stays false).
A scan leaves out ports where nothing answered, except the active endpoint and
an editor too busy to answer. An editor that accepts the connection but does
not answer within the timeout is busy: it is running a script, loading or
compiling. It is listed with `reachable` false and `error` saying so, and
`ping_unreal` and `unreal_health_check` report it with `busy: true` and
`error_type: InstanceBusy`. A
port whose probe never left this machine (out of file descriptors or local
ports, or a host name that does not resolve) is not dropped silently:
`warnings` names it, since an editor there would be missing.
The payload also carries `active_instance` (the active entry's name),
`total_discovered` (reachable instances) and `warnings`, which also report a
missing, invalid or stale attachment in attached mode.

With one identified editor and one viewport, selectors may be omitted. Unreal
layouts often contain several viewports: supply a key explicitly when selection
is ambiguous. These keys identify level viewports, not operating system windows
or arbitrary asset-editor tabs.

The selection is stored in `$SIMUL_HOME/unreal/attachment.json` (default `~/.simul/unreal/attachment.json`) with owner-only
permissions on POSIX. Passphrases are not copied into that file; configure
`UNREAL__PASSPHRASE` on the client when the editor requires authentication.

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
| `move_cursor` | `agent_id`, normalized `position: [x,y]` from bottom-left, optional `activity` label |
| `clear_cursor` | Remove only this `agent_id`'s marker in the attached viewport |

For actor actions, omit `target` to use the sole selected actor. They fail
explicitly, as the actor tools do, and change nothing: a `target` no actor has
is `NotFoundError`; a label several actors share is `ValidationError` with
`details.code: "ambiguous_target"` and their paths in `details.candidates`;
no `target` with other than one actor selected is `ValidationError` with
`details.code: "not_one_selected"`; a stale attachment is `InvalidState`.
Location uses centimeters;
rotation uses Pitch/Yaw/Roll in degrees and returns Unreal's normalized angles.
Transform editing participates in an editor undo transaction. While Play In
Editor runs, `inspect` and `clear_cursor` work (`selected_actors` is empty, as
Unreal reports no selection during PIE) and every other action returns
`error_type: "PIEActive"`; stop PIE with `control_unreal_simulation`, action
`stop`. The attachment check before each call finds the editor world during
PIE too, so the other Unreal tools keep working on the attached editor (see
[tools.md](tools.md#unreal-engine) for how each behaves while PIE runs).

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
operations use it as well, set `UNREAL__MODE=attached`. They then refuse `--host`
and `--port`, which name an endpoint attached mode never dials: the command exits 2
with `InvalidArgument` before contacting anything, as the JSON envelope under
`--json` and as a message on stderr otherwise. To work on another editor, attach
it. Existing MCP tools also use the selected endpoint when the server runs with
`--unreal-mode attached`. The default `endpoint` mode continues to use host/port
configuration.

## Visible agent cursors

For Blender-style pointers inside Unreal, close the project after saving your work
and install the bundled **Simul Agent Overlay** editor plugin through setup:

```sh
simul unreal setup /absolute/path/MyProject.uproject --agent-overlay --no-headless --yes
```

This compiles the plugin using that Unreal installation's C++ toolchain (Xcode on
macOS, Visual Studio on Windows, clang on Linux). Supply `--engine-path` if the
installation cannot be located. The source ships in the Python wheel; binaries are
built for your installed engine. Build failures report a log in the project's
`Saved/Logs`. Identical installations are reused; a different existing plugin is
preserved and setup explains how to move it aside before rebuilding.

Attach again after the editor restarts. Choose the visible viewport; `instances`
reports `active_viewport` as well as all layout keys, including hidden viewports.
Each concurrent agent should use a unique, stable `agent_id` on every call:

```sh
simul unreal control move_cursor --agent-id builder --position '[0.3,0.6]' --activity 'Placing scene actors'
simul unreal control move_cursor --agent-id reviewer --position '[0.7,0.4]' --activity 'Reviewing transforms'
simul unreal control select_actor --agent-id builder --target Cube
simul unreal control inspect
simul unreal control clear_cursor --agent-id reviewer
```

The overlay draws a colored pointer, agent name and activity row in the attached
level viewport. Named mutations update that agent's activity to the completed
action and preserve its last pointer position (center initially). `move_cursor`
only moves this virtual pointer; it does not click or move the OS mouse. Position
is an annotation, not automatic actor projection or a button selector.

Markers expire after 120 seconds without updates, are bounded to 64 per viewport,
and clear on map replacement, viewport destruction or editor shutdown. They live
only in Slate widgets, never in scene actors or saved assets, and cannot intercept
mouse input. Coincident pointers have separate activity rows. Scene screenshots
from `capture_unreal_viewport` exclude editor UI; use an editor-window screenshot
to see the overlay.

`inspect` returns `overlay_available` and all `agent_cursors` in the attached
viewport. Without the plugin, existing named controls still work and report
`overlay_available: false`; explicit cursor actions fail with the setup command.
Automatic labels cover named controls, not arbitrary scripts or all legacy tools.
Use `move_cursor` with an activity label around other work. Markers identify
callers; agent IDs do not isolate shared scene edits or actor selection.

## Identity checks and limits

The first discovery call (`simul unreal instances` or `list_unreal_instances`)
installs a small in-memory Python module and a map-change listener in each editor
that answers. It changes no scene data. Opening/replacing a map, restarting the editor,
or removing the selected viewport invalidates the selection. Re-run `instances`
and `attach`; a changed target is never accepted automatically. Reattaching or
detaching takes effect on the next tool session without restarting the MCP server.

Named controls and Python-backed tools check identity inside the same editor Python
execution as their operation, first, in the call's own namespace, which is discarded
with the call (every simul Python call runs in one; see
[tools.md](tools.md#unreal-engine)). Its runner lives in a second small in-memory
module, refreshed on every call. Legacy Remote Control calls check immediately before
dispatch in a separate request: attachment is not an editor lock, so avoid replacing
maps concurrently with those calls. Viewport config keys can be reused by Unreal
layouts; reattach after reorganizing level-editor windows. An explicit arbitrary
script can still change maps or perform other actions allowed by its execution policy.
Legacy camera and screenshot operations require the attached viewport to be active;
they refuse to use a different active viewport.

Named controls use Unreal's editor APIs. They do not inject OS mouse/keyboard input,
open arbitrary menus, address asset-editor tabs, or dismiss modal dialogs. A busy
editor can time out. Mutations are not automatically retried; a timeout after dispatch
means the operation may have executed, so inspect before retrying.

The editor serves Remote Control on its game thread, one request at a time and in
arrival order, and keeps running a request after simul stops waiting for it. A
request sent meanwhile would queue behind it, time out too and then run after it
was reported failed. So after a `TimeoutError` simul holds the editor, in either
mode: it sends it nothing more, and every Unreal call until the editor has
finished gets `InstanceBusy` at once, with `details.reason:
"finishing_timed_out_call"`, `timed_out_tool`, `changes_state` (whether that
call may still change the editor) and `busy_for_seconds`. Nothing answers
without the game thread, `GET /remote/info` included, so simul learns that the
editor is done from one `GET /remote/info` sent when the hold starts: its answer
means everything queued before it has run. `ping_unreal` and
`unreal_health_check` are still sent during the hold: while the editor is busy
they report it with `busy: true` and `held_after_timeout`, and an answer ends the
hold. An editor that stops taking connections (it quit or crashed) ends it too.
The hold lasts at most `UNREAL__HOLD_AFTER_TIMEOUT` seconds (default 300), after
which the log warns and later calls queue behind the request again; `0` turns it
off. A stopping server releases it at once. The hold covers every simul server on
the machine: it is also recorded under `$SIMUL_HOME/holds/` (the editor's
`host:port`, the timed-out call, the holding server's pid and an expiry), and
another simul server sends that editor nothing while the record is live, answering
`InstanceBusy` with `details.held_by` naming the holder's pid, until the holder
sees the editor answer and removes it. A record past its expiry is removed. A
server that stops while it holds the editor leaves its record `released`, and
one that crashed or was killed leaves it as it was (a holder counts as gone when
its pid has exited or, by the process start time the record keeps, names
another process): the next simul server about to send that editor a request
takes the hold over, sends its own `GET /remote/info` first, and goes ahead if
the editor answers within a second; otherwise the request gets `InstanceBusy`
(`details.adopted_from` names the server it came from) and that server holds the
editor until it answers, within the original limit. A record exists only once an
answer has timed out, so a request another server sends while the first one is
still running still queues behind it; nothing covers that. The servers must
share `SIMUL_HOME` (default `~/.simul`); another client of Remote Control is not
covered and still queues behind the request.

```sh
simul unreal detach
```

Detaching only forgets the selection. The editor and unsaved work remain open.
Use distinct `UNREAL__ATTACHMENT_PATH` values for clients targeting different editors.

## Verification

```sh
pytest tests/unreal/test_unreal_attachment.py --no-cov
# Only against a copied, disposable project: this test replaces the scratch map.
SIMUL_UNREAL_ATTACH_LIVE=1 UNREAL__PORT=30019 \
  pytest tests/unreal/test_live_attachment.py -m unreal_live --no-cov
# Add SIMUL_UNREAL_OVERLAY_LIVE=1 when setup installed the native overlay.
```

The live test exercises the named MCP actions with arbitrary scripting disabled,
checks dirty-map preservation on attach, rejects replaced-map targets and verifies
reattachment. Implemented against Unreal 5.7 editor APIs; other versions require
verification of their Python exposure before claiming compatibility.
