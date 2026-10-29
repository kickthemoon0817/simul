"""Unreal editor discovery and explicit attachment; no launcher or scene loading.

``discover_instances`` is the one discovery implementation: ``simul unreal
instances`` emits its payload, the ``list_unreal_instances`` MCP tool returns
it, and ``UnrealAttachments.attach`` selects from it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import Settings
from ..utils.private_files import BridgeFiles

# An attachment record is a few hundred bytes; refuse anything that is not.
MAX_ATTACHMENT_BYTES = 64 * 1024

# What an attachment pins besides its viewport; the editor's verify() checks these.
_TARGET_KEYS = ("instance_id", "document_id", "project_path", "map_path")
# Identity read through the editor's Python, null until it answers.
_IDENTITY_KEYS = (
    "instance_id",
    "document_id",
    "pid",
    "engine_version",
    "project_name",
    "project_path",
    "map_path",
    "active_viewport",
)
_IDENTITY_LISTS = ("viewports", "dirty_map_packages")


def read_attachment(path: Path) -> dict[str, Any]:
    """Read a private attachment record, refusing missing or malformed targets."""
    try:
        data = BridgeFiles.read(path, max_bytes=MAX_ATTACHMENT_BYTES)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No Unreal editor attached; run simul unreal instances, then simul unreal attach"
        ) from exc
    if data.get("version") != 1:
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


def _endpoint_settings(settings: Settings, host: str, port: int) -> Settings:
    """Settings for a plain endpoint session to one editor, whatever the mode."""
    cfg = settings.unreal.model_copy(
        update={"mode": "endpoint", "host": host, "port": port}
    )
    return settings.model_copy(update={"unreal": cfg})


async def probe_instance(settings: Settings, host: str, port: int) -> dict[str, Any]:
    """Probe one Remote Control endpoint and read its editor's identity.

    ``GET /remote/info`` decides ``reachable`` and ``latency_ms``; when it
    answers, the editor's Python reports the identity. An HTTP 401 also counts
    as an answer: Remote Control refuses a missing or wrong passphrase, so the
    editor is reachable without identity. Each step is bounded by
    ``unreal.ping_timeout``. A failed step does not raise: the identity fields
    stay null (lists empty) and ``error`` says why.

    Returns:
        ``host``, ``port``, ``reachable``, ``latency_ms``, ``error`` and the
        identity fields of ``UnrealRuntimeSession.editor_identity``.
    """
    from .unreal_runtime import UnrealRuntimeSession

    timeout = settings.unreal.ping_timeout
    record: dict[str, Any] = {
        "host": host,
        "port": port,
        "reachable": False,
        "latency_ms": None,
        "error": None,
        **{key: None for key in _IDENTITY_KEYS},
        **{key: [] for key in _IDENTITY_LISTS},
    }
    session = UnrealRuntimeSession(_endpoint_settings(settings, host, port))
    try:
        ping = await session.ping()
        if ping.get("status") == 401:
            # Remote Control answered, so an editor is there; listing it with
            # the reason beats a scan silently dropping it.
            record["reachable"] = True
            record["error"] = (
                "Remote Control refused the request (HTTP 401): set "
                "UNREAL__PASSPHRASE to the editor's passphrase"
            )
            return record
        if not ping.get("reachable"):
            record["error"] = ping.get("error") or f"No answer within {timeout}s"
            return record
        record["reachable"] = True
        record["latency_ms"] = ping.get("latency_ms")
        try:
            identity = await asyncio.wait_for(session.editor_identity(), timeout)
        except TimeoutError:
            record["error"] = (
                f"Editor identity did not answer within {timeout}s; the editor may "
                "be busy (UNREAL__PING_TIMEOUT sets the limit)"
            )
            return record
        except Exception as exc:
            record["error"] = (
                f"Editor identity unavailable ({exc}); discovery needs "
                "PythonScriptPlugin with remote Python execution (simul unreal setup)"
            )
            return record
        if identity.get("error"):
            record["error"] = f"Editor identity unavailable: {identity['error']}"
            return record
        for key in _IDENTITY_KEYS:
            record[key] = identity.get(key)
        for key in _IDENTITY_LISTS:
            record[key] = list(identity.get(key) or [])
        return record
    finally:
        await session.close()


def _is_attached(record: dict[str, Any], attachment: dict[str, Any] | None) -> bool:
    """Whether ``record`` is the attachment's editor, loaded map and viewport."""
    if attachment is None or record["port"] != attachment["port"]:
        return False
    target = attachment["target"]
    return (
        all(record[key] == target[key] for key in _TARGET_KEYS)
        and target["viewport"] in record["viewports"]
    )


async def discover_instances(
    settings: Settings,
    *,
    host: str | None = None,
    port: int | None = None,
    scan: bool = True,
) -> dict[str, Any]:
    """Discover running Unreal editors.

    The payload of both ``simul unreal instances`` and the
    ``list_unreal_instances`` MCP tool. Endpoints are probed concurrently with
    ``probe_instance``.

    The active endpoint is the one Unreal tools dispatch to: the configured
    ``unreal.host``/``unreal.port``, or the saved attachment's endpoint in
    attached mode. An instance is ``attached`` when it matches the saved
    attachment's editor, map and viewport, in either mode.

    Args:
        settings: Configuration; ``unreal`` supplies the host, ports, mode,
            timeouts, passphrase and attachment path.
        host: Host to probe instead of the active endpoint's host.
        port: Probe only this port.
        scan: Probe ``unreal.scan_port_start`` up to ``scan_port_end``
            (exclusive) plus the configured and attached ports. False probes
            only the active endpoint.

    Returns:
        ``success``, ``error``, ``instances`` (sorted by port),
        ``active_instance``, ``total_discovered`` (reachable instances) and
        ``warnings``. A scan leaves out ports where nothing answered, except
        the active endpoint; ``port`` and ``scan=False`` always report the
        endpoint they probed.
    """
    cfg = settings.unreal
    warnings: list[str] = []
    attachment: dict[str, Any] | None = None
    try:
        attachment = read_attachment(Path(cfg.attachment_path).expanduser())
    except (OSError, ValueError, RuntimeError) as exc:
        if cfg.mode == "attached":
            warnings.append(f"No usable Unreal attachment: {exc}")
    active: tuple[str, int] | None
    if cfg.mode == "attached":
        active = (attachment["host"], attachment["port"]) if attachment else None
    else:
        active = (cfg.host, cfg.port)

    probe_host = host or (active[0] if active else cfg.host)
    if port is not None:
        ports = [port]
    elif not scan:
        ports = [active[1]] if active else []
    else:
        extra = {cfg.port} | ({attachment["port"]} if attachment else set())
        ports = sorted(set(range(cfg.scan_port_start, cfg.scan_port_end)) | extra)
    report_all = port is not None or not scan

    records = await asyncio.gather(
        *(probe_instance(settings, probe_host, p) for p in ports)
    )
    instances: list[dict[str, Any]] = []
    for record in records:
        is_active = (record["host"], record["port"]) == active
        if not (record["reachable"] or is_active or report_all):
            continue
        instances.append(
            {
                "name": "default" if is_active else f"unreal-{record['port']}",
                "host": record["host"],
                "port": record["port"],
                "reachable": record["reachable"],
                "active": is_active,
                "attached": _is_attached(record, attachment),
                **{
                    key: value
                    for key, value in record.items()
                    if key not in ("host", "port", "reachable")
                },
            }
        )

    current = next((item for item in instances if item["active"]), None)
    if (
        cfg.mode == "attached"
        and current is not None
        and current["instance_id"]
        and not current["attached"]
    ):
        warnings.append(
            "The attached editor restarted, replaced its map or lost the attached "
            "viewport; run simul unreal instances, then simul unreal attach"
        )
    return {
        "success": True,
        "error": None,
        "instances": instances,
        "active_instance": current["name"] if current else None,
        "total_discovered": sum(1 for item in instances if item["reachable"]),
        "warnings": warnings,
    }


class UnrealAttachments:
    """Persist one explicitly selected editor/map/viewport from discovery."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = Path(settings.unreal.attachment_path).expanduser()

    async def describe(self, host: str, port: int) -> dict[str, Any]:
        """Probe one editor, raising when its identity cannot be read."""
        record = await probe_instance(self.settings, host, port)
        if not record["instance_id"]:
            raise RuntimeError(record["error"] or "Unreal editor identity unavailable")
        return record

    async def instances(
        self, host: str | None = None, port: int | None = None
    ) -> list[dict[str, Any]]:
        """Discovered instances (``discover_instances``) without selecting one."""
        found = await discover_instances(self.settings, host=host, port=port)
        return list(found["instances"])

    async def attach(
        self,
        host: str | None = None,
        port: int | None = None,
        viewport: str | None = None,
    ) -> dict[str, Any]:
        """Verify a unique target twice, then publish atomically without saving the map."""
        found = await self.instances(host, port)
        identified = [item for item in found if item.get("instance_id")]
        if len(identified) != 1:
            unidentified = "; ".join(
                f"{item['host']}:{item['port']}: {item['error']}"
                for item in found
                if item.get("reachable") and not item.get("instance_id")
            )
            raise ValueError(
                "No unique Unreal editor; run instances and select --host and --port. "
                "The existing editor needs RemoteControl and PythonScriptPlugin enabled."
                + (
                    f" Reachable without identity: {unidentified}"
                    if unidentified
                    else ""
                )
            )
        info = identified[0]
        views = info["viewports"]
        if viewport is None:
            if len(views) != 1:
                raise ValueError(
                    "No unique level viewport; select --viewport from instances"
                )
            viewport = views[0]
        if viewport not in views:
            raise ValueError("Unknown viewport; select --viewport from instances")
        target = {k: info[k] for k in _TARGET_KEYS}
        latest = await self.describe(info["host"], info["port"])
        if (
            any(latest[k] != v for k, v in target.items())
            or viewport not in latest["viewports"]
        ):
            raise RuntimeError(
                "Unreal changed while attaching; inspect instances and attach again"
            )
        target["viewport"] = viewport
        record = {
            "version": 1,
            "host": info["host"],
            "port": info["port"],
            "target": target,
        }
        BridgeFiles.write(self.path, record)
        return {
            **{key: value for key, value in latest.items() if key != "error"},
            "viewport": viewport,
            "mode": "attached",
            "attachment_path": str(self.path),
        }

    def detach(self) -> dict[str, Any]:
        """Forget the selection without closing or saving the editor."""
        self.path.unlink(missing_ok=True)
        return {"detached": True, "attachment_path": str(self.path)}
