"""Run authenticated operations in an explicitly selected Blender window.

The nonblocking socket is polled by an application timer. No worker thread
touches bpy, and every operation finishes before another can begin.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import secrets
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bpy
from bpy.app.handlers import persistent

from ..adapters.blender_runtime import BlenderRuntimeSession
from ..utils.paths import PathPolicy
from .agent_control import reset_ui
from .agent_cursor import cursors
from .protocol import MAX_MESSAGE_BYTES, PROTOCOL_VERSION, BridgeFiles, BridgeWire

logger = logging.getLogger(__name__)


@dataclass
class PendingConnection:
    """One bounded, short-lived connection polled on Blender's main thread."""

    socket: socket.socket
    expires: float
    incoming: bytearray = field(default_factory=bytearray)
    outgoing: bytes = b""


class BlenderBridge:
    """Identify a process, file-load generation, window, and scene before acting."""

    def __init__(self, discovery_dir: Path | None = None) -> None:
        self.discovery_dir = (
            discovery_dir
            or Path(
                os.environ.get("SIMUL_BLENDER_DISCOVERY_DIR", "~/.simul/blender")
            ).expanduser()
        )
        self.instance_id = uuid.uuid4().hex
        self.document_id = uuid.uuid4().hex
        self.token = secrets.token_urlsafe(32)
        self.listener: socket.socket | None = None
        self.connections: list[PendingConnection] = []
        self.discovery_path = self.discovery_dir / f"instance-{self.instance_id}.json"
        self._timer = self.poll
        self._load_handler = persistent(lambda unused: self._on_load(unused))

    def start(self) -> None:
        """Listen on loopback and advertise this instance without touching scene data."""
        if self.listener is not None:
            return
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
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
        """Read and execute bounded requests from Blender's main-thread timer."""
        if self.listener is None:
            return None
        cursors.prune()
        for _ in range(8):
            try:
                connection, _address = self.listener.accept()
            except BlockingIOError:
                break
            connection.setblocking(False)
            if len(self.connections) >= 8:
                connection.close()
                continue
            self.connections.append(PendingConnection(connection, time.monotonic() + 5))
        for pending in list(self.connections):
            try:
                if time.monotonic() > pending.expires:
                    self._close(pending)
                    continue
                if not pending.outgoing:
                    chunk = pending.socket.recv(65536)
                    if not chunk:
                        self._close(pending)
                        continue
                    pending.incoming.extend(chunk)
                    if len(pending.incoming) > MAX_MESSAGE_BYTES:
                        self._close(pending)
                        continue
                    if b"\n" not in pending.incoming:
                        continue
                    request = json.loads(pending.incoming.split(b"\n", 1)[0])
                    pending.outgoing = BridgeWire.encode(self.dispatch(request))
                    pending.expires = time.monotonic() + 5
                sent = pending.socket.send(pending.outgoing)
                pending.outgoing = pending.outgoing[sent:]
                if not pending.outgoing:
                    self._close(pending)
            except BlockingIOError:
                continue
            except (OSError, ValueError, TypeError):
                logger.exception("Blender bridge connection failed")
                self._close(pending)
        return 0.01 if self.connections else 0.05

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate identity before invoking a public session operation."""
        request_id = request.get("request_id") if isinstance(request, dict) else None
        try:
            if (
                not isinstance(request, dict)
                or request.get("protocol") != PROTOCOL_VERSION
            ):
                raise ValueError("Unsupported Blender bridge protocol")
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
                result = self._execute(request)
            return {"request_id": request_id, "success": True, "result": result}
        except Exception as exc:
            logger.error("Blender bridge request %s failed: %s", request_id, exc)
            return {
                "request_id": request_id,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "details": getattr(exc, "details", {}),
            }

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

    def _execute(self, request: dict[str, Any]) -> dict[str, Any]:
        target = request["target"]
        if target["document_id"] != self.document_id:
            raise ValueError(
                "Blender loaded another file; run simul blender attach again"
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
            raise ValueError(
                "Attached window closed or changed scene; run simul blender attach again"
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
        area = next((a for a in window.screen.areas if a.type == "VIEW_3D"), None)
        context = {"window": window}
        if area is not None:
            context["area"] = area
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is not None:
                context["region"] = region
        # Loading a file destroys the window/context; do not restore its stale
        # pointers on exiting a context override.
        if method == "open_blend_file":
            return dict(operation(*arguments.args, **arguments.kwargs))
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
            result = operation(*arguments.args, **arguments.kwargs)
            if method == "get_runtime_info":
                result.update(
                    instance_id=self.instance_id,
                    document_id=self.document_id,
                    window_id=target["window_id"],
                    scene_name=window.scene.name,
                    pid=os.getpid(),
                    is_dirty=bool(bpy.data.is_dirty),
                )
        return dict(result)

    def _on_load(self, _unused: Any) -> None:
        reset_ui()
        self.document_id = uuid.uuid4().hex

    def _close(self, pending: PendingConnection) -> None:
        pending.socket.close()
        if pending in self.connections:
            self.connections.remove(pending)
