"""Attach to a running Blender process without importing bpy or launching Blender."""

from __future__ import annotations

import inspect
import logging
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable

from ..blender_bridge.protocol import BridgeFiles, BridgeRemoteError, BridgeWire
from ..config import Settings
from ..resources import find_checkout_root
from ..utils.discovery import DiscoveryDir
from ..utils.paths import PathPolicy, SandboxDenied
from .blender_runtime import BlenderRuntimeSession, add_context_hint

logger = logging.getLogger(__name__)

REATTACH_HINT = (
    "call the attach_blender_window MCP tool (or run 'simul blender instances' "
    "and 'simul blender attach')"
)

# Bridge messages from add-ons built before the typed AttachmentTargetChanged
# error; matched so an older add-on still gets the structured recovery path.
_LEGACY_TARGET_CHANGED = (
    "Blender loaded another file",
    "Attached window closed or changed scene",
)


class AttachmentError(RuntimeError):
    """An attached-mode failure with a stable ``error_type`` clients can branch on."""

    error_type = "AttachmentError"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class AttachmentStale(AttachmentError):
    """The attached Blender process exited or its bridge stopped listening."""

    error_type = "AttachmentStale"


class AttachmentTargetChanged(AttachmentError):
    """The attached process is alive but loaded another file or lost the window/scene."""

    error_type = "AttachmentTargetChanged"


class BlenderConnection:
    """Session-compatible proxy pinned to a verified process, document and window."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def __getattr__(self, method: str) -> Callable[..., dict[str, Any]]:
        """Expose exactly the public session operations already used by MCP tools."""
        operation = getattr(BlenderRuntimeSession, method, None)
        if method.startswith("_") or method == "cleanup" or not callable(operation):
            raise AttributeError(method)

        def invoke(*args: Any, **kwargs: Any) -> dict[str, Any]:
            arguments = inspect.signature(operation).bind(None, *args, **kwargs)
            timeout = self.settings.blender.connection_timeout
            if method == "execute_script":
                if not self.settings.security.allow_script_execution:
                    raise PermissionError("Script execution is disabled")
                requested_timeout = arguments.arguments.get("timeout")
                if requested_timeout is not None:
                    timeout = min(timeout, float(requested_timeout))
            try:
                attachment = BridgeFiles.read(
                    Path(self.settings.blender.attachment_path).expanduser()
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "No Blender window attached; run simul blender instances, then simul blender attach"
                ) from exc
            policy = PathPolicy.from_settings(self.settings)
            payload = {
                "request_id": uuid.uuid4().hex,
                "instance_id": attachment["instance_id"],
                "target": attachment["target"],
                "deadline": time.time() + timeout,
                "method": method,
                "args": args,
                "kwargs": kwargs,
                "allow_script_execution": self.settings.security.allow_script_execution,
                "path_policy": {
                    "enabled": policy.enabled,
                    "allowed_paths": [str(p) for p in policy.allowed_roots],
                    "project_root": str(find_checkout_root() or Path.cwd()),
                    "allowed_url_schemes": policy.allowed_url_schemes,
                    "allowed_write_url_schemes": policy.allowed_write_url_schemes,
                },
            }
            self._refuse_dead_process(attachment)
            try:
                result = BridgeWire.request(attachment, payload, timeout)
            except ConnectionRefusedError as exc:
                # Refused while connecting: nothing was sent, so the operation
                # certainly did not run and naming the stale target is safe.
                raise self._stale(attachment, "its bridge refused the connection") from exc
            except BridgeRemoteError as exc:
                if exc.remote_type == "SandboxDenied":
                    raise SandboxDenied(
                        exc.details.get("file_path", ""), exc.details
                    ) from exc
                if exc.remote_type == "AttachmentTargetChanged" or any(
                    text in str(exc) for text in _LEGACY_TARGET_CHANGED
                ):
                    raise self._target_changed(attachment, exc) from exc
                raise
            except TimeoutError as exc:
                raise TimeoutError(
                    "Blender did not reply before the timeout. An operation that already started may still be "
                    "running; its outcome is unknown. No retry was sent. Inspect Blender before retrying."
                ) from exc
            if method == "execute_script" and result.get("error"):
                # Also covers add-ons built before the session added the hint itself.
                result = {**result, "error": add_context_hint(result["error"])}
            if method == "open_blend_file":
                result = {**result, **self._follow_document(attachment)}
            return result

        return invoke

    def _refuse_dead_process(self, attachment: dict[str, Any]) -> None:
        """Fail before connecting when the recorded process no longer exists.

        A dead process's port may since have been reused by another program,
        so probing the pid first also avoids talking to the wrong listener.
        """
        pid = attachment.get("pid")
        if type(pid) is int and pid > 0 and not DiscoveryDir.pid_alive(pid):
            raise self._stale(attachment, "the process is not running", alive=False)

    def _stale(
        self, attachment: dict[str, Any], reason: str, alive: bool | None = None
    ) -> AttachmentStale:
        """Explain a dead target with its pid and the recovery commands."""
        pid = attachment.get("pid")
        if alive is None and type(pid) is int and pid > 0:
            alive = DiscoveryDir.pid_alive(pid)
        who = f"Attached Blender (pid {pid})" if pid is not None else "Attached Blender"
        if alive is False:
            message = f"{who} is not running; {REATTACH_HINT} to select a live window."
        else:
            message = (
                f"{who} is not accepting bridge connections ({reason}); the Simul Blender "
                f"Bridge add-on was disabled or restarted. Re-enable it, then {REATTACH_HINT}."
            )
        return AttachmentStale(
            message,
            {
                "pid": pid,
                "process_alive": alive,
                "instance_id": attachment.get("instance_id"),
                "attachment_path": str(Path(self.settings.blender.attachment_path).expanduser()),
            },
        )

    def _target_changed(
        self, attachment: dict[str, Any], exc: BridgeRemoteError
    ) -> AttachmentTargetChanged:
        """Name what the pinned target became so the agent can re-attach deliberately."""
        previous = attachment.get("target", {})
        details: dict[str, Any] = {
            "attached_document_id": previous.get("document_id"),
            "attached_window_id": previous.get("window_id"),
            "instance_id": attachment.get("instance_id"),
        }
        if exc.details.get("document_id"):
            details["document_id"] = exc.details["document_id"]
        try:
            current = BlenderAttachments(self.settings)._hello(attachment)
            details["document_id"] = current["document_id"]
            details["blend_file_path"] = current.get("blend_file_path")
            details["windows"] = current.get("windows", [])
        except (OSError, ValueError, KeyError, RuntimeError):
            logger.debug("Could not describe the changed Blender target", exc_info=True)
        if details.get("document_id") not in (None, previous.get("document_id")):
            change = (
                f"Blender loaded another file (attached document {previous.get('document_id')}, "
                f"now {details['document_id']})"
            )
        else:
            change = "The attached Blender window closed or changed scene"
        return AttachmentTargetChanged(
            f"{change}. The attachment stays pinned to the old target so no call acts on an "
            f"unexpected scene; {REATTACH_HINT} to select the new one.",
            details,
        )

    def _follow_document(self, attachment: dict[str, Any]) -> dict[str, Any]:
        """Re-pin after this connection's own open_blend_file replaced the document.

        The open was requested through this attachment, so following it is not
        a silent retarget. Ambiguity (several windows, none of them the old one)
        is reported instead of guessed.
        """
        try:
            info = BlenderAttachments(self.settings).reattach(attachment)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            logger.warning("Could not re-attach after opening a file: %s", exc)
            return {
                "reattached": False,
                "reattach_error": f"{exc}; {REATTACH_HINT} to select a window.",
            }
        return {
            "reattached": True,
            "document_id": info["document_id"],
            "window_id": info["window"]["window_id"],
            "scene_name": info["window"]["scene_name"],
        }


class BlenderAttachments:
    """Discover and explicitly select windows served by the Blender add-on."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.directory = Path(settings.blender.discovery_dir).expanduser()
        self.attachment_path = Path(settings.blender.attachment_path).expanduser()

    def instances(self) -> list[dict[str, Any]]:
        """List live and stale bridge records, omitting authentication credentials."""
        results = []
        for path in sorted(self.directory.glob("instance-*.json")):
            try:
                endpoint = BridgeFiles.read(path)
                info = self._hello(endpoint)
                results.append({**info, "reachable": True})
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                results.append(
                    {
                        "instance_id": path.stem.removeprefix("instance-"),
                        "reachable": False,
                        "error": str(exc),
                    }
                )
        return results

    def attach(
        self, instance_id: str | None = None, window_id: str | None = None
    ) -> dict[str, Any]:
        """Verify a single explicit target, then atomically persist its identity."""
        live = [
            entry
            for entry in self.instances()
            if entry["reachable"] and not entry["background"]
        ]
        if instance_id is not None:
            live = [entry for entry in live if entry["instance_id"] == instance_id]
        if not live:
            raise RuntimeError(
                "No matching GUI Blender bridge. Enable the Simul Blender Bridge add-on in the existing window."
            )
        if len(live) != 1:
            raise ValueError(
                "Multiple Blender instances; select --instance from simul blender instances"
            )
        info = live[0]
        windows = info["windows"]
        if window_id is not None:
            windows = [window for window in windows if window["window_id"] == window_id]
        if len(windows) != 1:
            raise ValueError(
                "Select one --window from simul blender instances; no unique matching window"
            )
        endpoint = BridgeFiles.read(
            self.directory / f"instance-{info['instance_id']}.json"
        )
        return self._publish(endpoint, info, windows[0])

    def reattach(self, previous: dict[str, Any]) -> dict[str, Any]:
        """Re-pin the same process after its document changed.

        Keeps the previous window when it survived; otherwise requires the
        process to have exactly one window. Never switches processes.
        """
        instance_id = previous["instance_id"]
        endpoint = BridgeFiles.read(self.directory / f"instance-{instance_id}.json")
        info = self._hello(endpoint)
        old_window = previous.get("target", {}).get("window_id")
        windows = [
            window for window in info["windows"] if window["window_id"] == old_window
        ] or info["windows"]
        if len(windows) != 1:
            raise ValueError(
                f"Blender has {len(windows)} windows after the file changed; select one window_id"
            )
        return self._publish(endpoint, info, windows[0])

    def _publish(
        self, endpoint: dict[str, Any], info: dict[str, Any], selected: dict[str, Any]
    ) -> dict[str, Any]:
        """Recheck the target, then atomically persist it as the attachment."""
        # Recheck immediately before publishing, including the document and scene.
        latest = self._hello(endpoint)
        if (
            latest["document_id"] != info["document_id"]
            or selected not in latest["windows"]
        ):
            raise RuntimeError(
                "Blender changed while attaching; inspect instances and attach again"
            )
        target = {"document_id": info["document_id"], **selected}
        BridgeFiles.write(self.attachment_path, {**endpoint, "target": target})
        return {
            **info,
            "window": selected,
            "attachment_path": str(self.attachment_path),
            "mode": "attached",
        }

    def detach(self) -> dict[str, Any]:
        """Forget the target without touching Blender or its unsaved work."""
        self.attachment_path.unlink(missing_ok=True)
        return {"detached": True, "attachment_path": str(self.attachment_path)}

    def _hello(self, endpoint: dict[str, Any]) -> dict[str, Any]:
        timeout = min(self.settings.blender.connection_timeout, 2.0)
        return BridgeWire.request(
            endpoint,
            {
                "request_id": uuid.uuid4().hex,
                "instance_id": endpoint["instance_id"],
                "deadline": time.time() + timeout,
                "method": "hello",
            },
            timeout,
        )

    @staticmethod
    def build_addon(output: Path) -> Path:
        """Bundle the shared operations and policy without installing server dependencies into Blender."""
        package = Path(__file__).resolve().parents[1]
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        members = {
            "__init__.py": "blender_bridge/addon.py",
            "blender_bridge/__init__.py": "blender_bridge/__init__.py",
            "blender_bridge/bridge.py": "blender_bridge/bridge.py",
            "blender_bridge/agent_control.py": "blender_bridge/agent_control.py",
            "blender_bridge/agent_cursor.py": "blender_bridge/agent_cursor.py",
            "blender_bridge/protocol.py": "blender_bridge/protocol.py",
            "adapters/blender_runtime.py": "adapters/blender_runtime.py",
            "utils/paths.py": "utils/paths.py",
            "utils/private_files.py": "utils/private_files.py",
            "resources/__init__.py": "resources/__init__.py",
        }
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for destination, source in members.items():
                archive.write(package / source, f"simul_blender_bridge/{destination}")
            for subpackage in ("adapters", "utils"):
                archive.writestr(f"simul_blender_bridge/{subpackage}/__init__.py", "")
        return output
