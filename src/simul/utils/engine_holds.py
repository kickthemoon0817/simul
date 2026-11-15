"""Engine holds after a timed-out call, shared by every simul server on the machine.

Isaac Sim and Unreal's Remote Control go on running a request after its
client stopped waiting for it, and run the next request only after it. So a
server whose call timed out holds the engine: it sends it nothing more and
answers ``InstanceBusy`` until the engine has finished (the Isaac hold in
``SimulMCPServer._hold_if_engine_busy``, the Unreal one in
``simul.adapters._unreal_hold``). That hold lives in one process, and in the
usual multi-agent setup each agent's client spawns its own stdio server: a
second server on the same engine knew nothing of it, sent its own call, and
that call timed out too and ran after being reported failed (round-3 audit
M3-2).

So each hold is also published here, as one record file per hold under
``$SIMUL_HOME/holds/``, and every server checks the records for an engine
before it sends it anything:

- A record names the engine's endpoint, when the hold began, the call that
  timed out, the holding server's ``pid`` and ``host``, and when the hold
  expires (its start plus the hold limit).
- Only the server that wrote a record changes it: it writes the file
  atomically when the hold begins and removes it when the engine is idle
  again, the hold reaches its limit or the server stops. No lock is needed.
- A record past its expiry, or whose process on this host has exited (a
  crash, SIGKILL), is ignored, and removed by whoever sees it.
- A record of this process counts only while the hold that wrote it is still
  running (``_IN_PROCESS``), so a hold whose event loop ended without
  withdrawing it (a test, a CLI run) is not mistaken for a live one.

Reading and writing records is best effort: a hold that cannot be published
is logged and kept in its own server, and a directory that cannot be read
reads as "no records", which is how the servers behaved before.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .paths import state_dir
from .private_files import BridgeFiles

logger = logging.getLogger(__name__)

# Under $SIMUL_HOME (default ~/.simul); created on the first write, not here.
_DEFAULT_HOLD_DIR = state_dir() / "holds"
_HOST = socket.gethostname()
#: Largest record read; a record is a few hundred bytes.
MAX_RECORD_BYTES = 16 * 1024
#: Host names that all mean this machine, so two servers configured with
#: ``localhost`` and ``127.0.0.1`` find each other's holds.
_LOOPBACK_NAMES = frozenset({"localhost", "::1", "ip6-localhost", "0.0.0.0", "::"})
_UNSAFE = re.compile(r"[^A-Za-z0-9]+")

#: The holds this process published and has not withdrawn, by record file
#: name: each reports whether the hold that wrote it is still running.
_IN_PROCESS: Dict[str, Callable[[], bool]] = {}


def endpoint_key(host: str, port: int) -> str:
    """The ``host:port`` a hold is recorded under, the same however the host is spelled.

    Args:
        host: The engine's host as a server is configured with it.
        port: Its port (Isaac Sim's stock socket, Unreal's Remote Control).

    Returns:
        ``127.0.0.1:<port>`` for any loopback spelling, otherwise the host in
        lower case and the port.
    """
    name = str(host).strip().strip("[]").lower()
    if name in _LOOPBACK_NAMES or name.startswith("127."):
        name = "127.0.0.1"
    return f"{name}:{int(port)}"


def _pid_alive(pid: int) -> bool:
    """Whether process ``pid`` exists on this host (POSIX; True when it cannot be told)."""
    if sys.platform == "win32":  # pragma: no cover - no side-effect-free probe there
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


@dataclass(frozen=True)
class SharedHold:
    """A hold another simul server keeps on an engine, as its record says.

    Attributes:
        backend: ``isaac`` or ``unreal``.
        endpoint: The engine's ``endpoint_key``.
        tool: The call that timed out.
        since: ``time.time()`` when its reply timed out.
        expires_at: ``time.time()`` at which the hold ends whatever happens.
        pid: The holding server's process.
        host: The host it runs on.
        details: What else the backend records about the call (Unreal: its
            route and whether it may change the editor).
    """

    backend: str
    endpoint: str
    tool: str
    since: float
    expires_at: float
    pid: int
    host: str
    details: Tuple[Tuple[str, Any], ...] = ()

    def info(self) -> Dict[str, Any]:
        """Describe the hold as the refusals and status tools report it.

        Returns:
            ``timed_out_tool``, the backend's ``details``,
            ``busy_for_seconds``, ``hold_limit_seconds`` and ``held_by``
            (the other server's ``pid``).
        """
        return {
            "timed_out_tool": self.tool,
            **dict(self.details),
            "busy_for_seconds": round(max(0.0, time.time() - self.since), 1),
            "hold_limit_seconds": round(max(0.0, self.expires_at - self.since), 3),
            "held_by": {"pid": self.pid},
        }


def _as_hold(data: Dict[str, Any]) -> Optional[SharedHold]:
    """Read one parsed record, or None when it is not one."""
    details = data.get("details")
    try:
        hold = SharedHold(
            backend=str(data["backend"]),
            endpoint=str(data["endpoint"]),
            tool=str(data["tool"]),
            since=float(data["since"]),
            expires_at=float(data["expires_at"]),
            pid=int(data["pid"]),
            host=str(data["host"]),
            details=tuple(sorted(details.items())) if isinstance(details, dict) else (),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return hold


class SharedHolds:
    """The hold records of one backend: publishing this server's, reading the others'.

    One instance per hold keeper (the Isaac server, an Unreal adapter); its
    own records carry its token in their file name and never count as
    another server's.

    Args:
        backend: ``isaac`` or ``unreal``; records of other backends are
            never read.
        directory: Where the records live; ``$SIMUL_HOME/holds`` by default.
    """

    def __init__(self, backend: str, directory: Optional[Path] = None) -> None:
        self.backend = backend
        self._dir = directory if directory is not None else _DEFAULT_HOLD_DIR
        self._token = uuid.uuid4().hex[:12]
        self._published = 0

    @property
    def directory(self) -> Path:
        """The directory holding the records."""
        return self._dir

    def _prefix(self, endpoint: str) -> str:
        return f"{self.backend}-{_UNSAFE.sub('_', endpoint)}."

    def publish(
        self,
        endpoint: str,
        tool: str,
        since: float,
        limit: float,
        live: Callable[[], bool],
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Record a hold this server keeps, for the other servers on the machine.

        Args:
            endpoint: The engine's ``endpoint_key``.
            tool: The call that timed out.
            since: ``time.time()`` when its reply timed out.
            limit: The hold's limit in seconds; the record expires then.
            live: Whether the hold is still running; read while the record
                is checked from this process.
            details: JSON fields the other servers report with the hold.

        Returns:
            The record, for ``withdraw``; None when it could not be written
            (logged: the hold then covers this server only).
        """
        self._published += 1
        path = self._dir / f"{self._prefix(endpoint)}{os.getpid()}.{self._token}.{self._published}.json"
        record = {
            "backend": self.backend,
            "endpoint": endpoint,
            "tool": tool,
            "since": since,
            "expires_at": since + max(0.0, limit),
            "pid": os.getpid(),
            "host": _HOST,
        }
        if details:
            record["details"] = dict(details)
        try:
            BridgeFiles.write(path, record)
        except OSError as exc:
            logger.warning(
                "Could not record the hold on %s for the other simul servers (%s); "
                "it covers this server only",
                endpoint,
                exc,
            )
            return None
        _IN_PROCESS[path.name] = live
        return path

    def withdraw(self, record: Optional[Path]) -> None:
        """Remove a record ``publish`` wrote; nothing happens for None."""
        if record is None:
            return
        _IN_PROCESS.pop(record.name, None)
        try:
            record.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "Could not remove the hold record %s (%s); other servers ignore it once it "
                "expires or this process exits",
                record,
                exc,
            )

    def foreign(self, endpoint: str) -> Optional[SharedHold]:
        """The live hold another simul server keeps on ``endpoint``, or None.

        The oldest one when there are several. Records past their expiry or
        whose process has exited are removed on the way.

        Args:
            endpoint: The engine's ``endpoint_key``.
        """
        holds = self._live_records(endpoint)
        return min(holds, key=lambda hold: hold.since) if holds else None

    def _live_records(self, endpoint: str) -> List[SharedHold]:
        prefix = self._prefix(endpoint)
        try:
            names = [
                entry.name
                for entry in os.scandir(self._dir)
                if entry.name.startswith(prefix) and entry.name.endswith(".json")
            ]
        except OSError:
            return []  # no directory yet: nothing was ever held
        now = time.time()
        live: List[SharedHold] = []
        for name in names:
            parts = name[len(prefix):].split(".")
            if len(parts) != 4 or parts[1] == self._token:
                continue  # not a record, or one of this keeper's own
            path = self._dir / name
            try:
                data = BridgeFiles.read(path, max_bytes=MAX_RECORD_BYTES)
            except (OSError, ValueError):
                continue  # gone meanwhile, another user's, or damaged
            hold = _as_hold(data)
            if hold is None or hold.backend != self.backend or hold.endpoint != endpoint:
                continue
            if self._is_live(hold, name, now):
                live.append(hold)
            else:
                _IN_PROCESS.pop(name, None)
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        return live

    @staticmethod
    def _is_live(hold: SharedHold, name: str, now: float) -> bool:
        if now >= hold.expires_at:
            return False
        if hold.host != _HOST:
            return True  # another machine's process cannot be checked; it expires
        if hold.pid == os.getpid():
            running = _IN_PROCESS.get(name)
            try:
                return bool(running is not None and running())
            except Exception:  # pragma: no cover - a check that fails counts as ended
                return False
        return _pid_alive(hold.pid)
