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
  timed out, the holding server's ``pid``, ``host`` and process start time
  (``pid_started``), and when the hold expires (its start plus the hold
  limit).
- The server that wrote a record writes it atomically when the hold begins
  and removes it when the engine is idle again or the hold reaches its
  limit. A server that stops while it holds the engine marks its record
  ``released`` instead, since the engine may still be busy.
- A record past its expiry is removed by whoever sees it.
- A record whose server is gone (it exited, was killed, or another process
  now has its pid, as the start time shows) or that it ``released`` is an
  *orphan*: the engine may still be running the call. The next server that
  is about to send that engine something takes the hold over
  (``SharedHolds.adopt``): it publishes the record as its own, removes the
  orphan, and runs its own idle probe before it sends anything. Exactly one
  server takes an orphan over (the one whose unlink of it succeeds).
- A record of this process counts only while the hold that wrote it is still
  running (``_IN_PROCESS``), so a hold whose event loop ended without
  withdrawing it (a test) is not mistaken for a live one.
- A temporary file a writer killed mid-publish left behind is removed once
  it is ``STALE_TEMP_SECONDS`` old.

What a record cannot cover: it exists only once a reply has timed out, so a
call another server sends while the first call is still running (before its
timeout) still queues behind it inside the engine. Within one server the
instance lock prevents that; across servers, for Isaac Sim, enforced claims
do (``isaac_sim.enforce_claims``). Unreal has no equivalent.

Reading and writing records is best effort: a hold that cannot be published
is logged and kept in its own server, and a directory that cannot be read
reads as "no records", which is how the servers behaved before.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .paths import state_dir
from .private_files import BridgeFiles
from .process_identity import own_start, pid_alive, process_started

logger = logging.getLogger(__name__)

# Under $SIMUL_HOME (default ~/.simul); created on the first write, not here.
_DEFAULT_HOLD_DIR = state_dir() / "holds"
_HOST = socket.gethostname()
#: Largest record read; a record is a few hundred bytes.
MAX_RECORD_BYTES = 16 * 1024
#: Age at which a temporary file in the directory (``BridgeFiles.write``
#: names them ``.simul-*``) is taken for one a writer killed mid-publish left.
STALE_TEMP_SECONDS = 60.0
#: Host names that all mean this machine, so two servers configured with
#: ``localhost`` and ``127.0.0.1`` find each other's holds.
_LOOPBACK_NAMES = frozenset({"localhost", "::1", "ip6-localhost", "0.0.0.0", "::"})
_UNSAFE = re.compile(r"[^A-Za-z0-9]+")
_TEMP_PREFIX = ".simul-"

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


@dataclass(frozen=True)
class SharedHold:
    """A hold a simul server keeps or kept on an engine, as its record says.

    Attributes:
        backend: ``isaac`` or ``unreal``.
        endpoint: The engine's ``endpoint_key``.
        tool: The call that timed out.
        since: ``time.time()`` when its reply timed out.
        expires_at: ``time.time()`` at which the hold ends whatever happens.
        pid: The holding server's process.
        host: The host it runs on.
        details: What else the backend records about the call (Isaac: the
            transport; Unreal: its route and whether it may change the
            editor; ``adopted_from`` for a hold taken over).
        pid_started: The holding process's start time
            (``process_identity.process_started``); None in a record that
            has none.
        orphaned: The holder released the record as it stopped, or is gone:
            the next server to reach the engine takes the hold over.
        record: The record file.
    """

    backend: str
    endpoint: str
    tool: str
    since: float
    expires_at: float
    pid: int
    host: str
    details: Tuple[Tuple[str, Any], ...] = ()
    pid_started: Optional[str] = None
    orphaned: bool = False
    record: Optional[Path] = field(default=None, compare=False)

    def info(self) -> Dict[str, Any]:
        """Describe the hold as the refusals and status tools report it.

        Returns:
            ``timed_out_tool``, the backend's ``details``,
            ``busy_for_seconds``, ``hold_limit_seconds`` and ``held_by``
            (the other server's ``pid``, with ``stopped: true`` for an
            orphan: the next call checks the engine before it is sent).
        """
        holder: Dict[str, Any] = {"pid": self.pid}
        if self.orphaned:
            holder["stopped"] = True
        return {
            "timed_out_tool": self.tool,
            **{key: value for key, value in self.details if key != "transport"},
            "busy_for_seconds": round(max(0.0, time.time() - self.since), 1),
            "hold_limit_seconds": round(max(0.0, self.expires_at - self.since), 3),
            "held_by": holder,
        }

    def detail(self, key: str, default: Any = None) -> Any:
        """One of the backend's ``details``, or ``default``."""
        return dict(self.details).get(key, default)


def _as_hold(data: Dict[str, Any], record: Path) -> Optional[SharedHold]:
    """Read one parsed record, or None when it is not one."""
    details = data.get("details")
    started = data.get("pid_started")
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
            pid_started=started if isinstance(started, str) else None,
            record=record,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return hold


def _holder_gone(hold: SharedHold) -> bool:
    """Whether the process a record on this host names has exited, or its pid now names another process."""
    if not pid_alive(hold.pid):
        return True
    if hold.pid_started is None:
        return False  # a record without a start time: the pid is all there is
    now = process_started(hold.pid)
    return now is not None and now != hold.pid_started


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
        record: Dict[str, Any] = {
            "backend": self.backend,
            "endpoint": endpoint,
            "tool": tool,
            "since": since,
            "expires_at": since + max(0.0, limit),
            "pid": os.getpid(),
            "host": _HOST,
        }
        started = own_start()
        if started is not None:
            record["pid_started"] = started
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
        """Remove a record ``publish`` wrote (the engine is idle, or the hold reached its limit).

        Nothing happens for None.
        """
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

    def release(self, record: Optional[Path]) -> None:
        """Leave a hold to the other servers: this server stops while the engine may still be busy.

        The record stays, marked ``released``, so the next server about to
        send the engine something takes the hold over and checks the engine
        first (``adopt``). A record that cannot be marked stays as it is,
        and counts as an orphan once this process has exited. Nothing
        happens for None.
        """
        if record is None:
            return
        _IN_PROCESS.pop(record.name, None)
        try:
            data = BridgeFiles.read(record, max_bytes=MAX_RECORD_BYTES)
            data["released"] = True
            BridgeFiles.write(record, data)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            logger.warning(
                "Could not mark the hold record %s released (%s); the other servers take the hold "
                "over once this process has exited",
                record,
                exc,
            )

    def foreign(self, endpoint: str) -> Optional[SharedHold]:
        """The live hold another simul server keeps on ``endpoint``, or None.

        The oldest one when there are several. Expired records are removed on
        the way; orphans are left for ``orphan``.

        Args:
            endpoint: The engine's ``endpoint_key``.
        """
        live, _orphans = self.records(endpoint)
        return _oldest(live)

    def orphan(self, endpoint: str) -> Optional[SharedHold]:
        """The oldest orphaned hold on ``endpoint`` (its server released it or is gone), or None.

        Args:
            endpoint: The engine's ``endpoint_key``.
        """
        _live, orphans = self.records(endpoint)
        return _oldest(orphans)

    def adopt(self, orphan: SharedHold, live: Callable[[], bool]) -> Tuple[bool, Optional[Path]]:
        """Take over an orphaned hold, before this server sends the engine anything.

        The orphan is republished as this server's record, with the same
        call, start and expiry (``details.adopted_from`` names the server it
        came from), and then removed. Only the server whose removal succeeds
        keeps the hold; one that got there first had already published its
        own record, so the others see a live hold from then on.

        Args:
            orphan: A hold ``orphan`` returned.
            live: As for ``publish``.

        Returns:
            ``(taken, record)``: ``taken`` is False when another server took
            the hold over first (this server's copy is withdrawn again).
            ``record`` is this server's record; None when it could not be
            written, and then the hold covers this server only and the
            orphan stays for the others.
        """
        details = {key: value for key, value in orphan.details if key != "adopted_from"}
        details["adopted_from"] = orphan.pid
        mine = self.publish(
            orphan.endpoint, orphan.tool, orphan.since, orphan.expires_at - orphan.since, live, details
        )
        if mine is None or orphan.record is None:
            return True, mine
        try:
            os.unlink(orphan.record)
        except FileNotFoundError:
            self.withdraw(mine)
            return False, None
        except OSError:
            pass  # it stays: another server may take it over too, and probe as well
        return True, mine

    def records(self, endpoint: str) -> Tuple[List[SharedHold], List[SharedHold]]:
        """The other keepers' live and orphaned holds on ``endpoint``, from one read of the directory.

        Expired records, and this process's records whose hold has ended,
        are removed on the way.

        Args:
            endpoint: The engine's ``endpoint_key``.

        Returns:
            ``(live, orphans)``.
        """
        prefix = self._prefix(endpoint)
        names: List[str] = []
        now = time.time()
        try:
            with os.scandir(self._dir) as entries:
                for entry in entries:
                    if entry.name.startswith(prefix) and entry.name.endswith(".json"):
                        names.append(entry.name)
                    elif entry.name.startswith(_TEMP_PREFIX):
                        _remove_if_stale_temp(entry, now)
        except OSError:
            return [], []  # no directory yet: nothing was ever held
        live: List[SharedHold] = []
        orphans: List[SharedHold] = []
        for name in names:
            parts = name[len(prefix):].split(".")
            if len(parts) != 4 or parts[1] == self._token:
                continue  # not a record, or one of this keeper's own
            path = self._dir / name
            try:
                data = BridgeFiles.read(path, max_bytes=MAX_RECORD_BYTES)
            except (OSError, ValueError):
                continue  # gone meanwhile, another user's, or damaged
            hold = _as_hold(data, path)
            if hold is None or hold.backend != self.backend or hold.endpoint != endpoint:
                continue
            state = _state(hold, name, now, bool(data.get("released")))
            if state == "live":
                live.append(hold)
            elif state == "orphan":
                orphans.append(replace(hold, orphaned=True))
            else:
                _IN_PROCESS.pop(name, None)
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        return live, orphans


def _oldest(holds: List[SharedHold]) -> Optional[SharedHold]:
    """The hold that began first, or None."""
    return min(holds, key=lambda hold: hold.since) if holds else None


def _state(hold: SharedHold, name: str, now: float, released: bool) -> str:
    """``live``, ``orphan`` (released, or its server is gone) or ``stale`` (expired, or this process's ended hold)."""
    if now >= hold.expires_at:
        return "stale"
    if released:
        return "orphan"
    if hold.host != _HOST:
        return "live"  # another machine's process cannot be checked; it expires
    if hold.pid == os.getpid() and hold.pid_started in (None, own_start()):
        running = _IN_PROCESS.get(name)
        try:
            return "live" if running is not None and running() else "stale"
        except Exception:  # pragma: no cover - a check that fails counts as ended
            return "stale"
    return "orphan" if _holder_gone(hold) else "live"


def _remove_if_stale_temp(entry: "os.DirEntry[str]", now: float) -> None:
    """Remove a ``BridgeFiles.write`` temporary file of this user older than ``STALE_TEMP_SECONDS``."""
    try:
        info = entry.stat(follow_symlinks=False)
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            return
        if now - info.st_mtime >= STALE_TEMP_SECONDS:
            os.unlink(entry.path)
    except OSError:
        pass
