"""Explicit attachment to a running Unreal editor; no launcher or scene loading."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from ..config import Settings
from ._unreal_attach_scripts import EDITOR_STATE


def read_attachment(path: Path) -> dict[str, Any]:
    """Read a private attachment record, refusing missing or malformed targets."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError as exc:
        raise RuntimeError("No Unreal editor attached; run simul unreal instances, then simul unreal attach") from exc
    with os.fdopen(fd) as stream:
        meta = os.fstat(stream.fileno())
        if not stat.S_ISREG(meta.st_mode):
            raise ValueError("Unreal attachment must be a regular file")
        if hasattr(os, "getuid") and (meta.st_uid != os.getuid() or meta.st_mode & 0o077):
            raise PermissionError("Unreal attachment must be owned by this user and mode 0600")
        data = json.loads(stream.read(65537))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Invalid Unreal attachment record")
    if not isinstance(data.get("host"), str) or not data["host"]:
        raise ValueError("Invalid Unreal attachment host")
    if type(data.get("port")) is not int or not 1024 <= data["port"] <= 65535:
        raise ValueError("Invalid Unreal attachment port")
    target = data.get("target")
    if not isinstance(target, dict) or any(
        not isinstance(target.get(k), str) or not target[k]
        for k in ("instance_id", "document_id", "project_path", "map_path", "viewport")
    ):
        raise ValueError("Invalid Unreal attachment identity")
    return data


class UnrealAttachments:
    """Discover editors and persist one explicitly selected editor/map/viewport."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = Path(settings.unreal.attachment_path).expanduser()

    async def describe(self, host: str, port: int) -> dict[str, Any]:
        """Read editor identity using a temporary endpoint session."""
        from .unreal_runtime import UnrealRuntimeSession

        cfg = self.settings.unreal.model_copy(update={"mode": "endpoint", "host": host, "port": port})
        session = UnrealRuntimeSession(self.settings.model_copy(update={"unreal": cfg}))
        try:
            result = await session._execute_json_script(EDITOR_STATE + "print(json.dumps(_simul_editor.identity()))")
            if result.get("error"):
                raise RuntimeError(result["error"])
            return {**result, "host": host, "port": port, "reachable": True}
        finally:
            await session.close()

    async def instances(self, host: str | None = None, port: int | None = None) -> list[dict[str, Any]]:
        """Discover reachable editors on configured ports without selecting one."""
        cfg = self.settings.unreal
        host = host or cfg.host
        ports = [port] if port is not None else sorted({cfg.port, *range(cfg.scan_port_start, cfg.scan_port_end)})
        async def probe(p: int) -> dict[str, Any] | None:
            try:
                return await asyncio.wait_for(self.describe(host, p), timeout=cfg.ping_timeout)
            except Exception:
                return None
        results = await asyncio.gather(*(probe(p) for p in ports))
        return [item for item in results if item is not None]

    async def attach(self, host: str | None = None, port: int | None = None,
                     viewport: str | None = None) -> dict[str, Any]:
        """Verify a unique target twice, then publish atomically without saving the map."""
        found = await self.instances(host, port)
        if len(found) != 1:
            raise ValueError("No unique Unreal editor; run instances and select --host and --port. "
                             "The existing editor needs RemoteControl and PythonScriptPlugin enabled.")
        info = found[0]
        views = info["viewports"]
        if viewport is None:
            if len(views) != 1:
                raise ValueError("No unique level viewport; select --viewport from instances")
            viewport = views[0]
        if viewport not in views:
            raise ValueError("Unknown viewport; select --viewport from instances")
        target = {k: info[k] for k in ("instance_id", "document_id", "project_path", "map_path")}
        latest = await self.describe(info["host"], info["port"])
        if any(latest[k] != v for k, v in target.items()) or viewport not in latest["viewports"]:
            raise RuntimeError("Unreal changed while attaching; inspect instances and attach again")
        target["viewport"] = viewport
        record = {"version": 1, "host": info["host"], "port": info["port"], "target": target}
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(prefix=".simul-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(record, stream)
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {**info, "viewport": viewport, "mode": "attached", "attachment_path": str(self.path)}

    def detach(self) -> dict[str, Any]:
        """Forget the selection without closing or saving the editor."""
        self.path.unlink(missing_ok=True)
        return {"detached": True, "attachment_path": str(self.path)}
