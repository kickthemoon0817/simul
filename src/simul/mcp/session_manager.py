"""
Instance session manager for multi-agent coordination.

Tracks which agents are using which Isaac Sim instances via file-based
session records at ``~/.simul/sessions/``. Sessions are advisory by default —
agents are informed about who's doing what but not blocked. With
``isaac_sim.enforce_claims`` enabled the MCP server turns a live claim into a
lock: mutating tools from any other agent are refused until the claim is
released or expires after ``CLAIM_TTL_SECONDS`` without a heartbeat.

Every server process on the machine shares one record file per instance
port, so the file is the coordination point between them:

- Each read-modify-write holds an exclusive lock on ``{port}.lock`` (``flock``
  on POSIX, ``msvcrt.locking`` on Windows), so concurrent claims and
  heartbeats from several processes never lose each other's records.
- Each write goes to a temporary file that replaces the record file with
  ``os.replace``, so a reader never sees a half-written file.
- A record file that does not parse is repaired: the first complete record
  set it holds is kept when there is one, otherwise the file is moved aside to
  ``{port}.json.corrupt-<ms>``. Either way the store reports itself repaired
  for one claim TTL (``repaired_at``), and the server refuses mutating calls
  from agents that cannot show a claim of their own until it passes, because
  a claim the damaged file held may still be live.
- A store that cannot be read or written raises :class:`ClaimStoreError`
  instead of reading as "no claims" or reporting a claim that was never saved.

Each record names the server process that wrote it (``pid`` and ``host``) so a
claim whose process has exited, or whose MCP session ended, is recognised as
orphaned (:func:`claim_is_orphaned`): the server lets the same ``agent_id``
take it over at once instead of waiting out the TTL.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from ..utils.paths import state_dir

logger = logging.getLogger(__name__)

# Under $SIMUL_HOME (default ~/.simul); created on the first write, not here.
_DEFAULT_SESSION_DIR = state_dir() / "sessions"
CLAIM_TTL_SECONDS = 120.0
#: How long one store operation waits for another process's lock. The critical
#: sections are a file read and a write, so a wait this long means a process
#: holding the lock is stuck, and the operation fails instead of hanging.
LOCK_TIMEOUT_SECONDS = 5.0
_HOST = socket.gethostname()
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "for", "in", "on", "to", "of",
    "is", "it", "with", "from", "at", "by", "as", "this", "that",
    "isaac", "sim", "scene", "setup", "using", "test", "testing",
})

if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
    import msvcrt

    _LOCK_BUSY: tuple[type[BaseException], ...] = (OSError,)

    def _try_lock(fd: int) -> None:
        """Take the lock byte without waiting; raises OSError while another process holds it."""
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock(fd: int) -> None:
        """Release the lock byte."""
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    _LOCK_BUSY = (BlockingIOError,)

    def _try_lock(fd: int) -> None:
        """Take the exclusive lock without waiting; raises BlockingIOError while it is held."""
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        """Release the lock."""
        fcntl.flock(fd, fcntl.LOCK_UN)


class ClaimStoreError(Exception):
    """The claim records of an instance could not be read or written.

    Raised instead of reading as "no claims" (which would let claim
    enforcement fail open) and instead of reporting a claim that was never
    saved. ``error_type`` names it in a tool's error envelope.
    """

    error_type = "ClaimStoreError"

    def __init__(self, message: str, path: Path) -> None:
        super().__init__(message)
        self.path = path
        self.details: Dict[str, Any] = {"path": str(path)}


@dataclass
class ClaimView:
    """The live claims of one instance, as the store holds them.

    Attributes:
        sessions: The live session records.
        repaired_at: When the record file was last found damaged and repaired,
            while that is within one claim TTL; None otherwise.
    """

    sessions: List[Dict[str, Any]] = field(default_factory=list)
    repaired_at: Optional[float] = None
    changed: bool = False


#: Decides, with the store locked, whether a claim may be registered: returns
#: the record that blocks it, or None. See ``InstanceSession.register``.
ClaimConflict = Callable[[ClaimView], Optional[Dict[str, Any]]]


def _pid_alive(pid: int) -> bool:
    """Report whether process ``pid`` exists on this host (POSIX only)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def claim_is_orphaned(record: Dict[str, Any]) -> bool:
    """Report whether nothing can renew or release a claim any more.

    A claim is orphaned when the MCP session that made it ended (the server
    marks its records ``orphaned``) or when the server process that wrote it,
    on this host, has exited. Windows has no side-effect-free process probe in
    the standard library (``os.kill(pid, 0)`` sends CTRL_C there), so only the
    session mark counts on it.

    Args:
        record: One session record.

    Returns:
        True when the claim is orphaned.
    """
    if record.get("orphaned"):
        return True
    pid = record.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or record.get("host") != _HOST:
        return False
    if pid == os.getpid() or sys.platform == "win32":
        return False
    return not _pid_alive(pid)


@contextmanager
def _exclusive_lock(lock_path: Path, *, required: bool) -> Iterator[None]:
    """Hold the exclusive cross-process lock on ``lock_path``.

    Args:
        lock_path: The lock file, created on first use and never removed
            (removing a lock file races with a process about to lock it).
        required: Raise when the lock file cannot be opened. A read passes
            False: in a directory it may not write it reads unlocked, which is
            safe because writers replace the record file atomically.

    Raises:
        ClaimStoreError: The lock file could not be opened (``required``), or
            another process held the lock for ``LOCK_TIMEOUT_SECONDS``.
    """
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        if required:
            raise ClaimStoreError(
                f"Cannot open the claim lock {lock_path}: {exc}", lock_path
            ) from exc
        yield
        return
    try:
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
        delay = 0.0005
        while True:
            try:
                _try_lock(fd)
                break
            except _LOCK_BUSY:
                if time.monotonic() >= deadline:
                    raise ClaimStoreError(
                        f"Another process held the claim lock {lock_path} for "
                        f"{LOCK_TIMEOUT_SECONDS:g} s",
                        lock_path,
                    ) from None
                time.sleep(delay)
                delay = min(delay * 2, 0.01)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def _as_view(data: Any) -> Optional[ClaimView]:
    """Read a parsed record file, or None when it is not one."""
    if not isinstance(data, dict):
        return None
    sessions = data.get("sessions", [])
    if not isinstance(sessions, list) or not all(isinstance(s, dict) for s in sessions):
        return None
    repaired_at = data.get("repaired_at")
    if not isinstance(repaired_at, (int, float)) or isinstance(repaired_at, bool):
        repaired_at = None
    return ClaimView(sessions=list(sessions), repaired_at=repaired_at)


def _last_active(record: Dict[str, Any]) -> float:
    value = record.get("last_active", 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


class InstanceSession:
    """
    Manages session records for a single Isaac Sim instance.

    Each instance (identified by port) has a JSON file at
    ``~/.simul/sessions/{port}.json`` containing a list of active
    sessions with their purpose, agent ID, and heartbeat timestamp, guarded
    by ``{port}.lock``. See the module docstring for the concurrency rules.
    """

    def __init__(self, port: int, session_dir: Path | None = None) -> None:
        self._port = port
        self._dir = session_dir or _DEFAULT_SESSION_DIR
        self._file = self._dir / f"{port}.json"
        self._lock_file = self._dir / f"{port}.lock"

    # ------------------------------------------------------------------
    # Store primitives (call with the lock held)
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        """Create the session directory, private to the user, before a write."""
        try:
            self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise ClaimStoreError(
                f"Cannot create the session directory {self._dir}: {exc}", self._dir
            ) from exc

    def _load(self) -> ClaimView:
        """Read the record file, repairing it when it does not parse.

        Raises:
            ClaimStoreError: The file exists but cannot be read, or is damaged
                and cannot be moved aside.
        """
        try:
            raw = self._file.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ClaimView()
        except (OSError, UnicodeDecodeError) as exc:
            raise ClaimStoreError(
                f"Cannot read the claim records {self._file}: {exc}", self._file
            ) from exc
        try:
            view = _as_view(json.loads(raw))
        except json.JSONDecodeError:
            view = None
        if view is not None:
            return view
        return self._repair(raw)

    def _repair(self, raw: str) -> ClaimView:
        """Recover from a record file that does not parse.

        A file damaged by an interrupted or interleaved write usually starts
        with one complete record set; that one is kept. Anything else is moved
        aside. Both mark the store repaired, so claim enforcement refuses
        agents without a claim of their own for one TTL: a claim the damaged
        file held may still be live, and its holder re-asserts it meanwhile.
        """
        now = time.time()
        salvaged: Optional[ClaimView] = None
        try:
            data, _end = json.JSONDecoder().raw_decode(raw.lstrip())
            salvaged = _as_view(data)
        except json.JSONDecodeError:
            salvaged = None
        if salvaged is not None:
            logger.warning(
                "Claim records %s were damaged; kept the first complete record set "
                "(%d claims). Claim enforcement refuses mutating calls from agents "
                "without a claim on port %d for %d s while holders re-assert theirs.",
                self._file,
                len(salvaged.sessions),
                self._port,
                int(CLAIM_TTL_SECONDS),
            )
            salvaged.repaired_at = now
            salvaged.changed = True
            return salvaged
        quarantine = self._file.with_name(f"{self._file.name}.corrupt-{int(now * 1000)}")
        try:
            os.replace(self._file, quarantine)
        except OSError as exc:
            raise ClaimStoreError(
                f"The claim records {self._file} are damaged and could not be moved "
                f"aside: {exc}",
                self._file,
            ) from exc
        logger.warning(
            "Claim records %s were unreadable and were moved to %s. Claim enforcement "
            "refuses mutating calls from agents without a claim on port %d for %d s "
            "while holders re-assert theirs.",
            self._file,
            quarantine,
            self._port,
            int(CLAIM_TTL_SECONDS),
        )
        return ClaimView(repaired_at=now, changed=True)

    def _save(self, view: ClaimView) -> None:
        """Replace the record file with ``view`` in one atomic step.

        An empty store with no repair to report removes the file instead.

        Raises:
            ClaimStoreError: The file could not be written or replaced.
        """
        if not view.sessions and view.repaired_at is None:
            try:
                self._file.unlink(missing_ok=True)
            except OSError as exc:
                raise ClaimStoreError(
                    f"Cannot remove the claim records {self._file}: {exc}", self._file
                ) from exc
            return
        self._ensure_dir()
        document: Dict[str, Any] = {"port": self._port, "sessions": view.sessions}
        if view.repaired_at is not None:
            document["repaired_at"] = view.repaired_at
        temp_path: Optional[str] = None
        try:
            fd, temp_path = tempfile.mkstemp(
                dir=self._dir, prefix=f".{self._port}.", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2)
            os.replace(temp_path, self._file)
            temp_path = None
        except OSError as exc:
            raise ClaimStoreError(
                f"Cannot write the claim records {self._file}: {exc}", self._file
            ) from exc
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    @staticmethod
    def _prune(view: ClaimView, now: float) -> None:
        """Drop expired claims and a repair older than one TTL from ``view``."""
        ttl = CLAIM_TTL_SECONDS
        live = [s for s in view.sessions if now - _last_active(s) < ttl]
        if len(live) != len(view.sessions):
            view.sessions = live
            view.changed = True
        if view.repaired_at is not None and now - view.repaired_at >= ttl:
            view.repaired_at = None
            view.changed = True

    @contextmanager
    def _open(self, *, write: bool) -> Iterator[ClaimView]:
        """Lock the store and yield its live claims; saved on exit when changed.

        Args:
            write: The caller changes the store. A write failure then raises;
                for a read, writing back pruned records is housekeeping, and
                its failure is only logged.

        Raises:
            ClaimStoreError: The store could not be locked, read or, for a
                write, saved.
        """
        if write:
            self._ensure_dir()
        with _exclusive_lock(self._lock_file, required=write):
            view = self._load()
            self._prune(view, time.time())
            yield view
            if view.changed:
                try:
                    self._save(view)
                except ClaimStoreError as exc:
                    if write:
                        raise
                    logger.debug("Could not write back pruned claims for port %d: %s", self._port, exc)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def view(self) -> ClaimView:
        """Return the live claims and any recent repair.

        Raises:
            ClaimStoreError: The records could not be read.
        """
        with self._open(write=False) as view:
            return ClaimView(sessions=[dict(s) for s in view.sessions], repaired_at=view.repaired_at)

    def register(
        self,
        agent_id: str,
        purpose: str,
        owner: Optional[str] = None,
        *,
        conflict: Optional[ClaimConflict] = None,
    ) -> dict[str, Any]:
        """
        Register an agent session on this instance.

        Args:
            agent_id: Unique identifier for the agent/session.
            purpose: Free-text description of what the agent is doing.
            owner: Opaque token for the MCP session holding the claim. Claim
                   enforcement compares it as well as ``agent_id``, because
                   ``agent_id`` is caller-chosen and listed to every agent.
            conflict: Called with the live claims while the store is locked;
                      a record it returns blocks the registration. Checking
                      and registering under one lock is what keeps two
                      processes from both seeing a free instance and both
                      claiming it.

        Returns:
            ``status`` ``registered``, ``updated`` or ``reclaimed`` (an
            existing record of ``agent_id`` now held by a different
            ``owner``), or ``refused`` with the blocking record as ``holder``.

        Raises:
            ClaimStoreError: The records could not be read or saved; the claim
                was not registered.
        """
        with self._open(write=True) as view:
            if conflict is not None:
                holder = conflict(view)
                if holder is not None:
                    return {"status": "refused", "holder": dict(holder), "port": self._port}
            now = time.time()
            existing = [s for s in view.sessions if s.get("agent_id") == agent_id]
            if existing:
                record = existing[0]
                status = (
                    "reclaimed"
                    if owner is not None and record.get("owner") not in (None, owner)
                    else "updated"
                )
            else:
                record = {"agent_id": agent_id, "tools_used": [], "started": now}
                view.sessions.append(record)
                status = "registered"
            record["purpose"] = purpose
            record["last_active"] = now
            if owner is not None:
                record["owner"] = owner
            record["pid"] = os.getpid()
            record["host"] = _HOST
            record.pop("orphaned", None)
            view.changed = True
        return {"status": status, "agent_id": agent_id, "port": self._port}

    def heartbeat(self, agent_id: str, tool_name: str | None = None) -> bool:
        """
        Update the heartbeat timestamp for an agent session.

        Args:
            agent_id: The agent's session identifier.
            tool_name: Optional tool name to add to tools_used list.

        Returns:
            True when a live record of ``agent_id`` was refreshed.

        Raises:
            ClaimStoreError: The records could not be read or saved.
        """
        with self._open(write=True) as view:
            for s in view.sessions:
                if s.get("agent_id") != agent_id:
                    continue
                s["last_active"] = time.time()
                tools = s.get("tools_used")
                if not isinstance(tools, list):
                    tools = []
                if tool_name and tool_name not in tools:
                    tools = (tools + [tool_name])[-20:]
                s["tools_used"] = tools
                view.changed = True
                return True
        return False

    def release(
        self,
        agent_id: str,
        *,
        allowed: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> dict[str, Any]:
        """
        Release an agent's session on this instance.

        Args:
            agent_id: The agent's session identifier.
            allowed: Called with the agent's record while the store is
                locked; when it returns False the record is kept and the
                release is refused.

        Returns:
            ``status`` ``released`` or ``not_found``, or ``refused`` with the
            record as ``holder``.

        Raises:
            ClaimStoreError: The records could not be read or saved; nothing
                was released.
        """
        with self._open(write=True) as view:
            matching = [s for s in view.sessions if s.get("agent_id") == agent_id]
            if matching and allowed is not None and not allowed(matching[0]):
                return {"status": "refused", "holder": dict(matching[0]), "port": self._port}
            view.sessions = [s for s in view.sessions if s.get("agent_id") != agent_id]
            if matching:
                view.changed = True
        return {"status": "released" if matching else "not_found", "port": self._port}

    def mark_orphaned(self, owner: str) -> int:
        """Mark the claims ``owner`` holds as orphaned, once its MCP session has ended.

        The claims stay live until they expire, so another agent does not take
        the instance from one that is reconnecting, but the same ``agent_id``
        may take them over at once (see :func:`claim_is_orphaned`).

        Args:
            owner: The opaque owner token of the ended session.

        Returns:
            How many records were marked.

        Raises:
            ClaimStoreError: The records could not be read or saved.
        """
        marked = 0
        with self._open(write=True) as view:
            for s in view.sessions:
                if s.get("owner") == owner and not s.get("orphaned"):
                    s["orphaned"] = True
                    marked += 1
            if marked:
                view.changed = True
        return marked

    def clear(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Remove every live claim on this instance, or only ``agent_id``'s.

        An operator action (``simul isaac claims --clear``); it also ends a
        repair window, since the operator has decided what the claims are.

        Args:
            agent_id: Remove only this agent's claim.

        Returns:
            The removed records.

        Raises:
            ClaimStoreError: The records could not be read or saved.
        """
        removed: List[Dict[str, Any]] = []
        kept: List[Dict[str, Any]] = []
        with self._open(write=True) as view:
            for record in view.sessions:
                matches = agent_id is None or record.get("agent_id") == agent_id
                (removed if matches else kept).append(record)
            view.sessions = kept
            if agent_id is None:
                view.repaired_at = None
            view.changed = True
        return removed

    def get_status(self) -> dict[str, Any]:
        """
        Get the current session status for this instance.

        Returns:
            Dict with session count, active sessions, and overall status
            (``in_use``, ``free``, or ``unverified`` during a repair window
            with no claim re-asserted yet), plus ``repaired_at`` during a
            repair window.

        Raises:
            ClaimStoreError: The records could not be read.
        """
        view = self.view()
        status: Dict[str, Any] = {
            "port": self._port,
            "session_count": len(view.sessions),
            "status": "in_use" if view.sessions else (
                "unverified" if view.repaired_at is not None else "free"
            ),
            "sessions": view.sessions,
        }
        if view.repaired_at is not None:
            status["repaired_at"] = view.repaired_at
        return status


class SessionManager:
    """
    Manages instance sessions across all Isaac Sim instances.

    Provides discovery with compatibility scoring so agents can make
    informed decisions about which instance to use.
    """

    def __init__(self, session_dir: Path | None = None) -> None:
        # Created on the first write, so a home directory that cannot be
        # written does not stop the server from starting; the claim tools
        # report the failure instead.
        self._dir = session_dir or _DEFAULT_SESSION_DIR

    @property
    def session_dir(self) -> Path:
        """The directory holding the record files."""
        return self._dir

    def get_instance_session(self, port: int) -> InstanceSession:
        """
        Get the session manager for a specific instance port.

        Args:
            port: The TCP port of the Isaac Sim instance.

        Returns:
            InstanceSession for that port.
        """
        return InstanceSession(port, self._dir)

    def _ports(self) -> List[int]:
        """Return the ports that have a record file, in order."""
        ports: List[int] = []
        try:
            for f in sorted(self._dir.glob("*.json")):
                try:
                    ports.append(int(f.stem))
                except ValueError:
                    continue
        except OSError:
            pass
        return ports

    def get_all_sessions(self) -> list[dict[str, Any]]:
        """
        Read session status for all tracked instances.

        Returns:
            List of per-instance status dicts with live claims (or a repair
            window). An instance whose records cannot be read is listed with
            an ``error`` and no sessions, rather than left out.
        """
        results = []
        for port in self._ports():
            try:
                status = InstanceSession(port, self._dir).get_status()
            except ClaimStoreError as exc:
                logger.warning("Cannot read the claims for port %d: %s", port, exc)
                results.append(
                    {"port": port, "session_count": 0, "status": "unreadable",
                     "sessions": [], "error": str(exc)}
                )
                continue
            if status["sessions"] or "repaired_at" in status:
                results.append(status)
        return results

    def clear(
        self, port: Optional[int] = None, agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Remove live claims: every one, one instance's, or one agent's.

        Args:
            port: Limit to the instance on this port.
            agent_id: Limit to this agent's claims.

        Returns:
            The removed records, each with its ``port``.

        Raises:
            ClaimStoreError: An instance's records could not be read or saved.
        """
        removed: List[Dict[str, Any]] = []
        ports = [port] if port is not None else self._ports()
        for each in ports:
            for record in InstanceSession(each, self._dir).clear(agent_id):
                removed.append({"port": each, **record})
        return removed

    def score_compatibility(
        self,
        incoming_purpose: str,
        instance_port: int,
        status: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Score how compatible an incoming agent's purpose is with an instance's
        current sessions.

        Args:
            incoming_purpose: The new agent's purpose description.
            instance_port: Port of the instance to check.
            status: Session status already read for this instance. Passing it
                    avoids re-reading and re-parsing the same session file,
                    which callers that have just called ``get_status`` would
                    otherwise do twice per instance.

        Returns:
            Dict with compatibility level, score, reason, and existing sessions.
        """
        if status is None:
            status = InstanceSession(instance_port, self._dir).get_status()
        sessions = status["sessions"]

        if not sessions:
            return {
                "port": instance_port,
                "compatibility": "clear",
                "score": 1.0,
                "reason": "No active sessions — instance is free",
                "sessions": [],
            }

        incoming_tokens = _tokenize(incoming_purpose)
        best_score = 0.0
        # With no overlap at all the reason still has to name whose work the
        # caller would be interfering with, so start from the first session.
        best_match_purpose = sessions[0].get("purpose", "")

        for s in sessions:
            existing_tokens = _tokenize(s.get("purpose", ""))
            if not incoming_tokens or not existing_tokens:
                continue
            shared = incoming_tokens & existing_tokens
            union = incoming_tokens | existing_tokens
            score = len(shared) / len(union) if union else 0.0
            if score > best_score:
                best_score = score
                best_match_purpose = s.get("purpose", "")

        if best_score > 0.5:
            level = "compatible"
            reason = f"Similar purpose: \"{best_match_purpose}\" — safe to join"
        elif best_score > 0.0:
            level = "likely_safe"
            reason = f"Some overlap with: \"{best_match_purpose}\" — different but may coexist"
        else:
            level = "caution"
            reason = f"Different purpose from: \"{best_match_purpose}\" — may cause interference"

        return {
            "port": instance_port,
            "compatibility": level,
            "score": round(best_score, 3),
            "reason": reason,
            "sessions": sessions,
        }

    def cleanup_stale(self) -> int:
        """
        Remove all stale session files.

        Returns:
            Number of instances whose expired records were cleaned up.
        """
        cleaned = 0
        for port in self._ports():
            path = self._dir / f"{port}.json"
            try:
                InstanceSession(port, self._dir).get_status()
            except ClaimStoreError:
                continue
            # Reading prunes expired records and removes a file left empty.
            if not path.exists():
                cleaned += 1
        return cleaned


def _tokenize(text: str) -> set[str]:
    """
    Tokenize a purpose string into lowercase keywords, excluding stopwords.

    Args:
        text: Free-text purpose description.

    Returns:
        Set of meaningful lowercase tokens.
    """
    tokens = set()
    for word in text.lower().split():
        cleaned = "".join(c for c in word if c.isalnum() or c == "_")
        if cleaned and len(cleaned) > 1 and cleaned not in _STOPWORDS:
            tokens.add(cleaned)
    return tokens
