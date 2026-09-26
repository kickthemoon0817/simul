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
from ..utils.paths import PathPolicy, SandboxDenied
from .blender_runtime import BlenderRuntimeSession

logger = logging.getLogger(__name__)


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
            try:
                return BridgeWire.request(attachment, payload, timeout)
            except BridgeRemoteError as exc:
                if exc.remote_type == "SandboxDenied":
                    raise SandboxDenied(
                        exc.details.get("file_path", ""), exc.details
                    ) from exc
                raise
            except TimeoutError as exc:
                raise TimeoutError(
                    "Blender did not reply before the timeout. An operation that already started may still be "
                    "running; its outcome is unknown. No retry was sent. Inspect Blender before retrying."
                ) from exc

        return invoke


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
        selected = windows[0]
        endpoint = BridgeFiles.read(
            self.directory / f"instance-{info['instance_id']}.json"
        )
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
