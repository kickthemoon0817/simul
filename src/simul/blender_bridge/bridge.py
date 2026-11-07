"""Run authenticated operations in an explicitly selected Blender window.

The nonblocking socket is polled by an application timer. No worker thread
touches bpy, and every operation finishes before another can begin: requests
that arrive together wait their turn, oldest first, one operation per poll so
that Blender redraws and handles the user's input between them.
"""

from __future__ import annotations

import inspect
import itertools
import json
import logging
import os
import secrets
import socket
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import bpy
from bpy.app.handlers import persistent

from ..adapters.blender_runtime import BlenderRuntimeSession
from ..utils.errors import exception_message
from ..utils.paths import PathPolicy, state_dir
from .agent_control import reset_ui
from .agent_cursor import cursors, hide_annotations, observations
from .protocol import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    START_ACK,
    BridgeFiles,
    BridgeWire,
    protocol_mismatch_message,
)

logger = logging.getLogger(__name__)

#: Connections held at once, waiting or being answered. A further one is closed
#: unread, so its request certainly did not run, and a client of an add-on with
#: START_ACK sends it again.
MAX_CONNECTIONS = 64
#: Seconds a connection may go without sending or reading a byte while the
#: bridge polls. Time Blender's main thread spends elsewhere (an operation, a
#: render, a modal dialog) does not count: the bridge could not read then.
IDLE_SECONDS = 10.0
#: A gap between polls longer than this was Blender being busy, not the client.
STALL_SECONDS = 0.25
#: Seconds to keep reading and discarding a refused request's remaining bytes,
#: so that a client still sending it reads the refusal instead of a reset.
LINGER_SECONDS = 5.0
#: Bytes read from one connection per recv call.
READ_CHUNK = 1 << 20

#: Operations that change the open document, by the MCP tool that calls them.
#: Each completed call is recorded as one undo step named ``simul: <tool>``.
#: ``execute_script`` is recorded even when the script reports an error, since
#: it may have changed data before raising.
UNDO_STEPS: dict[str, str] = {
    "add_force_field": "add_blender_force_field",
    "add_modifier": "add_blender_modifier",
    "add_rigid_body_constraint": "add_blender_rigid_body_constraint",
    "apply_simready_metadata": "apply_simready_metadata",
    "assign_material": "assign_blender_material",
    "bake_simulation": "bake_blender_simulation",
    "clear_object_parent": "clear_blender_object_parent",
    "create_mesh_from_data": "create_blender_mesh_from_data",
    "create_object": "create_blender_object",
    "delete_keyframe": "delete_blender_keyframe",
    "delete_object": "delete_blender_object",
    "execute_script": "execute_blender_script",
    # It selects the objects it exports while it runs. Should the selection
    # stay changed, the step records it; otherwise the step costs nothing.
    "export_simready_usd": "export_simready_usd",
    "focus_on_object": "focus_blender_on_object",
    "free_bake": "free_blender_bake",
    "import_file": "import_blender_file",
    "insert_keyframe": "insert_blender_keyframe",
    "set_camera_view": "set_blender_camera_view",
    # Blender records frame changes as undo steps too.
    "set_frame": "set_blender_frame",
    "set_frame_range": "set_blender_frame_range",
    "set_light_params": "set_blender_light_params",
    "set_object_parent": "set_blender_object_parent",
    "set_object_transform": "set_blender_object_transform",
    "setup_rigid_body": "setup_blender_rigid_body",
    "setup_simready_hierarchy": "setup_simready_hierarchy",
}
#: ``control_ui`` actions that change scene data: the selection and the active
#: object's transform, which Blender's own selection and property edits record.
UNDO_UI_ACTIONS = frozenset({"select_object", "set_property"})
#: Operations recorded as no undo step and that leave the modified flag alone.
NO_UNDO_STEP = frozenset(
    {
        # Reads, including captures and samplers that restore the frame and
        # render settings they change.
        "capture_viewport",
        "capture_viewport_sequence",
        "get_bounding_box",
        "get_camera_info",
        "get_constraint_info",
        "get_distance_between",
        "get_file_info",
        "get_force_field_info",
        "get_frame",
        "get_keyframes",
        "get_material_info",
        "get_mesh_info",
        "get_object_info",
        "get_object_trajectory",
        "get_physics_state",
        "get_runtime_info",
        "get_simready_metadata",
        "get_viewport_info",
        "is_object_within_bounds",
        "list_scene_objects",
        "search_objects",
        "summarize_scene",
        "validate_simready_compliance",
        # Loading a file starts a new undo history; a step after saving would
        # mark the saved file modified again.
        "open_blend_file",
        "save_blend_file",
        # Writes only the exported file.
        "export_file",
        # Playback, which Blender does not record either.
        "play_animation",
        # Recorded per action: UNDO_UI_ACTIONS.
        "control_ui",
    }
)


def undo_step_name(method: str, arguments: dict[str, Any]) -> str | None:
    """Name the undo step a completed operation is recorded as, or None for none.

    An operation in neither table is recorded under its own name: an
    unrecorded edit can be lost, a spare step cannot.

    Args:
        method: The ``BlenderRuntimeSession`` operation.
        arguments: Its bound arguments.

    Returns:
        ``simul: <tool>``, or None when the operation changes no document data.
    """
    if method == "control_ui":
        action = arguments.get("agent_control")
        if action not in UNDO_UI_ACTIONS:
            return None
        return f"simul: control_blender_ui ({action})"
    if method in NO_UNDO_STEP:
        return None
    return f"simul: {UNDO_STEPS.get(method, method)}"


class AttachmentTargetChanged(ValueError):
    """The pinned document, window or scene is gone; the reply carries the new identity."""

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


class RequestAbandoned(ConnectionError):
    """The client disconnected before its request started, so it did not run."""


@dataclass
class PendingConnection:
    """One connection polled on Blender's main thread, from request to reply.

    Attributes:
        socket: The accepted, nonblocking socket.
        expires: Monotonic time at which a connection that is still sending its
            request, or reading its reply, is dropped; moved on by every byte
            and by the time Blender's main thread was busy elsewhere.
        incoming: The request bytes read so far.
        outgoing: Reply bytes not yet sent.
        request: The decoded request once its line is complete.
        ready: Whether ``request`` waits for its turn.
        order: Arrival order among waiting requests.
        discarding: A refused request whose remaining bytes are read and dropped.
    """

    socket: socket.socket
    expires: float
    incoming: bytearray = field(default_factory=bytearray)
    outgoing: bytes = b""
    request: Any = None
    ready: bool = False
    order: int = 0
    discarding: bool = False


class BlenderBridge:
    """Identify a process, file-load generation, window, and scene before acting."""

    def __init__(self, discovery_dir: Path | None = None) -> None:
        # The server looks in blender.discovery_dir, which defaults to
        # $SIMUL_HOME/blender: Blender has to see the same SIMUL_HOME.
        self.discovery_dir = discovery_dir or (
            Path(os.environ["SIMUL_BLENDER_DISCOVERY_DIR"]).expanduser()
            if os.environ.get("SIMUL_BLENDER_DISCOVERY_DIR")
            else state_dir() / "blender"
        )
        self.instance_id = uuid.uuid4().hex
        self.document_id = uuid.uuid4().hex
        self.token = secrets.token_urlsafe(32)
        self.listener: socket.socket | None = None
        self.connections: list[PendingConnection] = []
        self._arrivals = itertools.count()
        self._last_poll = 0.0
        # An undo step that waits for the next poll; see _push_undo_step.
        self.deferred_undo_step: str | None = None
        self.discovery_path = self.discovery_dir / f"instance-{self.instance_id}.json"
        self._timer = self.poll
        self._load_handler = persistent(lambda unused: self._on_load(unused))

    def start(self) -> None:
        """Listen on loopback and advertise this instance without touching scene data."""
        if self.listener is not None:
            return
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        # Requests that arrive while Blender is busy wait in this backlog.
        listener.listen(socket.SOMAXCONN)
        listener.setblocking(False)
        self.listener = listener
        try:
            BridgeFiles.write(
                self.discovery_path,
                {
                    "protocol": PROTOCOL_VERSION,
                    "instance_id": self.instance_id,
                    "pid": os.getpid(),
                    "host": "127.0.0.1",
                    "port": listener.getsockname()[1],
                    "token": self.token,
                    "capabilities": [START_ACK],
                },
            )
            bpy.app.handlers.load_post.append(self._load_handler)
            bpy.app.timers.register(self._timer, first_interval=0.05, persistent=True)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Detach transport resources; never quit Blender or save its data."""
        reset_ui()
        if bpy.app.timers.is_registered(self._timer):
            bpy.app.timers.unregister(self._timer)
        if self._load_handler in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.remove(self._load_handler)
        for pending in self.connections:
            pending.socket.close()
        self.connections.clear()
        if self.listener is not None:
            self.listener.close()
            self.listener = None
        self.discovery_path.unlink(missing_ok=True)

    def poll(self) -> float | None:
        """Accept, read and answer requests from Blender's main-thread timer.

        Never raises: Blender unregisters a timer whose call raises, which
        would stop the bridge.
        """
        if self.listener is None:
            return None
        try:
            self._poll_once()
        except BaseException:
            logger.exception("Blender bridge poll failed")
        return 0.01 if self.connections else 0.05

    def _poll_once(self) -> None:
        stalled = time.monotonic() - self._last_poll
        if self._last_poll and stalled > STALL_SECONDS:
            for pending in self.connections:
                pending.expires += stalled
        if self.deferred_undo_step is not None:
            step, self.deferred_undo_step = self.deferred_undo_step, None
            self._push_undo_step(step, "")
        cursors.prune()
        observations.prune()
        self._accept()
        for pending in list(self.connections):
            try:
                self._progress(pending)
            except OSError:
                self._close(pending)
        # Measured before any operation runs, so that the next poll counts its
        # time as Blender being busy.
        self._last_poll = time.monotonic()
        # Answer the oldest waiting requests until one runs an operation; the
        # rest wait for the next poll, after Blender has handled its events.
        while True:
            waiting = [p for p in self.connections if p.ready]
            if not waiting or self._answer(min(waiting, key=lambda p: p.order)):
                break

    def _accept(self) -> None:
        assert self.listener is not None
        for _ in range(MAX_CONNECTIONS):
            try:
                connection, _address = self.listener.accept()
            except BlockingIOError:
                return
            except OSError:
                logger.exception("Blender bridge could not accept a connection")
                return
            if len(self.connections) >= MAX_CONNECTIONS:
                # Unread, so it certainly did not run; an acknowledging client
                # sends it again.
                connection.close()
                continue
            connection.setblocking(False)
            self.connections.append(
                PendingConnection(connection, time.monotonic() + IDLE_SECONDS)
            )

    def _progress(self, pending: PendingConnection) -> None:
        """Move one connection's bytes as far as its socket allows without blocking."""
        if pending.discarding:
            self._send(pending)
            if self._drained(pending) or time.monotonic() > pending.expires:
                self._close(pending)
            return
        if pending.ready:
            if self._client_gone(pending):
                # It gave up while the request waited: nothing ran or will.
                self._close(pending)
            return
        if pending.request is None:
            self._read(pending)
        else:
            self._send(pending)
        if pending in self.connections and time.monotonic() > pending.expires:
            self._close(pending)

    def _read(self, pending: PendingConnection) -> None:
        while pending.request is None and not pending.discarding and pending in self.connections:
            try:
                chunk = pending.socket.recv(READ_CHUNK)
            except BlockingIOError:
                return
            if not chunk:
                # Closed before sending a whole request: nothing to run.
                self._close(pending)
                return
            pending.expires = time.monotonic() + IDLE_SECONDS
            pending.incoming.extend(chunk)
            if b"\n" in chunk:
                self._parse(pending)
            elif len(pending.incoming) > MAX_MESSAGE_BYTES:
                self._refuse_too_large(pending)

    def _parse(self, pending: PendingConnection) -> None:
        line, _, _ = bytes(pending.incoming).partition(b"\n")
        pending.incoming = bytearray()
        if len(line) + 1 > MAX_MESSAGE_BYTES:
            self._refuse_too_large(pending)
            return
        try:
            pending.request = json.loads(line)
        except (ValueError, RecursionError):
            self._refuse(
                pending,
                "InvalidRequest",
                "Blender bridge request is not valid JSON; it was not executed.",
                {},
            )
            return
        pending.ready = True
        pending.order = next(self._arrivals)

    def _answer(self, pending: PendingConnection) -> bool:
        """Dispatch a waiting request and start sending its reply.

        Returns:
            Whether it started an operation.
        """
        pending.ready = False
        started = False

        def start() -> None:
            nonlocal started
            self._acknowledge_start(pending)
            started = True

        reply = self.dispatch(pending.request, start)
        pending.outgoing += self._encode_reply(pending.request, reply)
        pending.expires = time.monotonic() + IDLE_SECONDS
        try:
            self._send(pending)
        except OSError:
            self._close(pending)
        return started

    def _acknowledge_start(self, pending: PendingConnection) -> None:
        """Confirm the client still waits, and acknowledge the start if it asked.

        Raises:
            RequestAbandoned: The client is gone or the acknowledgement could
                not be sent, so the operation must not run.
        """
        if self._client_gone(pending):
            raise RequestAbandoned(
                "The client disconnected before Blender started the request; it was not executed"
            )
        request = pending.request
        if request.get("ack") is not True:
            return
        message = BridgeWire.encode({"request_id": request.get("request_id"), "started": True})
        try:
            sent = pending.socket.send(message)
        except OSError as exc:
            raise RequestAbandoned(
                "Blender could not acknowledge the request's start; it was not executed"
            ) from exc
        # A few dozen bytes into an empty send buffer go whole; any rest goes
        # out ahead of the reply.
        pending.outgoing = message[sent:]

    @staticmethod
    def _client_gone(pending: PendingConnection) -> bool:
        try:
            return pending.socket.recv(1, socket.MSG_PEEK) == b""
        except BlockingIOError:
            return False
        except OSError:
            return True

    def _send(self, pending: PendingConnection) -> None:
        """Send what the socket takes; close an answered connection once its reply is out."""
        while pending.outgoing:
            try:
                sent = pending.socket.send(pending.outgoing)
            except BlockingIOError:
                return
            pending.outgoing = pending.outgoing[sent:]
            pending.expires = time.monotonic() + IDLE_SECONDS
        if not pending.discarding:
            self._close(pending)

    @staticmethod
    def _drained(pending: PendingConnection) -> bool:
        """Read and drop a refused request's bytes; True once the client closed.

        At most MAX_MESSAGE_BYTES per poll, so that a client that keeps sending
        cannot hold up the others.
        """
        for _ in range(max(1, MAX_MESSAGE_BYTES // READ_CHUNK)):
            try:
                chunk = pending.socket.recv(READ_CHUNK)
            except BlockingIOError:
                return False
            if not chunk:
                return True
        return False

    def _refuse_too_large(self, pending: PendingConnection) -> None:
        limit = MAX_MESSAGE_BYTES // 2**20
        self._refuse(
            pending,
            "RequestTooLarge",
            f"Blender bridge request exceeds {limit} MiB, the most the bridge reads per "
            "call; it was not executed. Send less per call, for example a large mesh as "
            "several objects.",
            {"limit_bytes": MAX_MESSAGE_BYTES},
        )

    def _refuse(
        self, pending: PendingConnection, error_type: str, message: str, details: dict[str, Any]
    ) -> None:
        """Answer a request the bridge will not run, without resetting the connection.

        Its client may still be sending; closing now would bury the refusal
        under a connection reset, so the remaining bytes are read and dropped
        for up to LINGER_SECONDS while the refusal goes out.
        """
        pending.incoming = bytearray()
        pending.request = None
        pending.discarding = True
        pending.expires = time.monotonic() + LINGER_SECONDS
        pending.outgoing = BridgeWire.encode(
            {
                "request_id": None,
                "success": False,
                "error": f"{error_type}: {message}",
                "error_type": error_type,
                "details": {**details, "executed": False},
            }
        )
        self._send(pending)

    def dispatch(
        self, request: dict[str, Any], on_start: Callable[[], None] | None = None
    ) -> dict[str, Any]:
        """Validate identity before invoking a public session operation.

        Args:
            request: The decoded request.
            on_start: Called right before the operation runs, after every check
                that can refuse the request; it may raise to keep it from
                running (the client left).

        Returns:
            The reply.
        """
        request_id = request.get("request_id") if isinstance(request, dict) else None
        try:
            if not isinstance(request, dict):
                raise ValueError("Unsupported Blender bridge protocol")
            if request.get("protocol") != PROTOCOL_VERSION:
                raise ValueError(
                    protocol_mismatch_message(PROTOCOL_VERSION, request.get("protocol"))
                )
            token = request.get("token")
            if not isinstance(token, str) or not secrets.compare_digest(
                token, self.token
            ):
                raise PermissionError("Blender bridge authentication failed")
            if request.get("instance_id") != self.instance_id:
                raise ValueError(
                    "Blender instance changed; run simul blender attach again"
                )
            if time.time() > float(request["deadline"]):
                raise TimeoutError(
                    "Request expired before execution; no operation was performed"
                )
            if request["method"] == "hello":
                result = self.describe()
            else:
                result = self._execute(request, on_start)
            return {"request_id": request_id, "success": True, "result": result}
        except BaseException as exc:
            # SystemExit and KeyboardInterrupt too: nothing above this timer
            # handles them. Blender prints an exception that escapes a timer
            # with PyErr_Print, which exits the process on SystemExit, and
            # unregisters the timer, which would stop the bridge, on any other.
            message = exception_message(exc)
            logger.error("Blender bridge request %s failed: %s", request_id, message)
            return {
                "request_id": request_id,
                "success": False,
                # Notes carry context such as the object a bpy operator was
                # adding a rigid body to; str(exc) alone would drop it.
                "error": f"{type(exc).__name__}: {message}",
                "error_type": type(exc).__name__,
                "details": getattr(exc, "details", {}),
            }

    @staticmethod
    def _encode_reply(request: Any, response: dict[str, Any]) -> bytes:
        """Encode a reply; an unencodable result still gets a small error reply.

        Closing the socket without a reply would leave the client with a bare
        connection error. The operation may already have run, so the error
        says so instead of inviting a blind retry.
        """
        try:
            return BridgeWire.encode(response)
        except (ValueError, TypeError) as exc:
            logger.error("Blender bridge reply could not be encoded: %s", exc)
            request_id = request.get("request_id") if isinstance(request, dict) else None
            return BridgeWire.encode(
                {
                    "request_id": request_id,
                    "success": False,
                    "error": (
                        "Blender handled the request, but its reply could not be encoded "
                        f"({exc}); the operation may already have run. Request a smaller "
                        "result (fewer frames, lower resolution) or inspect Blender before "
                        "retrying."
                    ),
                    "error_type": "ResponseEncodingError",
                    "details": {},
                }
            )

    def describe(self) -> dict[str, Any]:
        """Report live windows and unsaved state without choosing a target."""
        return {
            "instance_id": self.instance_id,
            "document_id": self.document_id,
            "pid": os.getpid(),
            "version_string": bpy.app.version_string,
            "blend_file_path": bpy.data.filepath or None,
            "is_dirty": bool(bpy.data.is_dirty),
            "background": bool(bpy.app.background),
            "windows": [
                {
                    "window_id": str(window.as_pointer()),
                    "scene_id": str(window.scene.as_pointer()),
                    "scene_name": window.scene.name,
                    "workspace": window.workspace.name,
                }
                for window in bpy.context.window_manager.windows
            ],
        }

    def _execute(
        self, request: dict[str, Any], on_start: Callable[[], None] | None = None
    ) -> dict[str, Any]:
        def start() -> None:
            # Checked again: the request may have waited behind others.
            if time.time() > float(request["deadline"]):
                raise TimeoutError(
                    "Request expired before execution; no operation was performed"
                )
            if on_start is not None:
                on_start()

        target = request["target"]
        if target["document_id"] != self.document_id:
            raise AttachmentTargetChanged(
                f"Blender loaded another file (attached document {target['document_id']}, "
                f"now {self.document_id}); call attach_blender_window or run simul blender attach again",
                {"document_id": self.document_id, "attached_document_id": target["document_id"]},
            )
        window = next(
            (
                w
                for w in bpy.context.window_manager.windows
                if str(w.as_pointer()) == target["window_id"]
            ),
            None,
        )
        if window is None or str(window.scene.as_pointer()) != target["scene_id"]:
            raise AttachmentTargetChanged(
                "Attached window closed or changed scene; call attach_blender_window "
                "or run simul blender attach again",
                {"document_id": self.document_id},
            )
        method = request["method"]
        operation = getattr(BlenderRuntimeSession, method, None)
        if method.startswith("_") or method == "cleanup" or not callable(operation):
            raise ValueError(f"Unsupported Blender operation: {method}")
        if method == "execute_script" and not request.get(
            "allow_script_execution", False
        ):
            raise PermissionError("Script execution is disabled")
        policy = dict(request["path_policy"])
        policy["project_root"] = Path(policy["project_root"])
        session = BlenderRuntimeSession(
            path_policy=PathPolicy(**policy), scene_scope=True
        )
        arguments = inspect.signature(operation).bind(
            session, *request.get("args", []), **request.get("kwargs", {})
        )
        if method == "execute_script":
            # Attached bpy must run inline on the main thread. The client bounds
            # waiting; it cannot safely interrupt an operation that already began.
            arguments.arguments["timeout"] = None
        # The largest 3D View is the one the user is most likely looking at:
        # viewport captures draw it and the capture cue marks it.
        area = max(
            (a for a in window.screen.areas if a.type == "VIEW_3D"),
            key=lambda a: a.width * a.height,
            default=None,
        )
        # Blender derives screen, scene and view layer from the window. Do not
        # pin scene/view_layer here: a script that loads a file (read_homefile,
        # open_mainfile) frees them, and a later operator in the same script
        # dereferences the stale pointer and crashes Blender (verified on
        # 5.0.1). With only window/area/region pinned, such a script gets a
        # missing-context-member AttributeError instead, which execute_script
        # annotates with the re-override recipe (issue #215).
        context = {"window": window}
        if area is not None:
            context["area"] = area
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is not None:
                context["region"] = region
        # Loading a file destroys the window/context; do not restore its stale
        # pointers on exiting a context override.
        if method == "open_blend_file":
            start()
            return dict(operation(*arguments.args, **arguments.kwargs))
        document_id = self.document_id
        with bpy.context.temp_override(**context):
            # A primitive-add operator in Edit Mode edits the user's active mesh
            # instead of creating an object. Require an explicit mode change.
            if (
                method
                in {
                    "create_object",
                    "delete_object",
                    "setup_rigid_body",
                    "add_force_field",
                    "add_rigid_body_constraint",
                    "add_modifier",
                    "import_file",
                    "create_mesh_from_data",
                }
                and bpy.context.mode != "OBJECT"
            ):
                raise ValueError(
                    "Attached window is not in Object Mode; leave Edit/Pose Mode before this operation"
                )
            capturing = method in {"capture_viewport", "capture_viewport_sequence"}
            # The label only drives cosmetic feedback. The MCP request models
            # validate it for both modes; an odd label here must not cost the
            # capture, so observations.show() rejects it inside the guard below.
            agent_id = arguments.arguments.get("agent_id", "agent")
            start()
            with hide_annotations() if capturing else nullcontext():
                result = operation(*arguments.args, **arguments.kwargs)
            if (
                capturing
                and area is not None
                and (result.get("image_base64") or result.get("frame_count", 0) > 0)
            ):
                # The image is complete before feedback appears in the editor.
                # Cosmetic feedback must never turn a completed capture into an error.
                try:
                    observations.show(window, area, agent_id)
                except Exception:
                    logger.exception("Could not display Blender capture feedback")
            if method == "get_runtime_info":
                result.update(
                    instance_id=self.instance_id,
                    document_id=self.document_id,
                    window_id=target["window_id"],
                    scene_name=window.scene.name,
                    pid=os.getpid(),
                    is_dirty=bool(bpy.data.is_dirty),
                )
        step = undo_step_name(method, arguments.arguments)
        if step is not None and self.document_id != document_id:
            # A script loaded a file. Blender marks the loaded file unmodified
            # when its event loop handles the load's notifier, after this timer
            # returns, so a step recorded now would leave what the script did
            # after the load counted as saved. The next poll records it.
            self.deferred_undo_step = step
        elif step is not None:
            self._push_undo_step(step, target["window_id"])
        return dict(result)

    def _push_undo_step(self, name: str, window_id: str) -> None:
        """Record a completed operation as one undo step, marking the file modified.

        Blender records an operator the user runs as an undo step once it
        finishes, but neither ``bpy.ops`` called from Python nor an RNA write
        records one. An agent's edits then joined whatever step the user's next
        action recorded, so one Ctrl+Z took back all of them, and the file never
        counted as modified, so quitting asked nothing. ``ed.undo_push`` records
        the step and also marks the file modified (``WM_file_tag_modified``).

        A script that loaded a file is recorded too, in the new file's history,
        on the next poll (``deferred_undo_step``): what it did after the load
        would otherwise be in no step and count as saved.

        Args:
            name: The step's name in Edit > Undo History.
            window_id: The attached window, which gives the operator its context
                while it is open; any window serves, the undo history being
                global, and a loaded file has new windows.
        """
        windows = list(bpy.context.window_manager.windows)
        window = next(
            (w for w in windows if str(w.as_pointer()) == window_id),
            windows[0] if windows else None,
        )
        if window is None:
            logger.error("No Blender window to record the undo step %r", name)
            return
        try:
            with bpy.context.temp_override(window=window):
                bpy.ops.ed.undo_push(message=name)
        except Exception:
            # The operation itself completed; failing it now would invite a retry.
            logger.exception("Blender did not record the undo step %r", name)

    def _on_load(self, _unused: Any) -> None:
        reset_ui()
        self.document_id = uuid.uuid4().hex

    def _close(self, pending: PendingConnection) -> None:
        pending.socket.close()
        if pending in self.connections:
            self.connections.remove(pending)
