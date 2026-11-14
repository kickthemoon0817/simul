"""Holding an Unreal editor after a request timed out, until the editor has finished it.

Remote Control serves every request on the editor's game thread, one at a
time and in the order they arrive, and goes on running a request after its
client stopped waiting for it. A request sent meanwhile waits behind it, so
it can time out too and then run after it was reported failed (round-3 audit
R3-M4: with a 5 s timeout, a marker script sent after a 12 s script timed out
and wrote its file a second after it was reported failed).

So after a timeout simul holds the editor: it sends it nothing more and
answers ``InstanceBusy`` until the editor has finished the request, for at
most ``unreal.hold_after_timeout`` seconds. No probe can tell without the
game thread: ``GET /remote/info`` is served on it like every other route, and
the editor's listening socket accepts connections while the game thread is
blocked. The probe is therefore that GET, sent once when the hold starts and
read until it answers, which proves that everything queued before it has run.

Each hold is also recorded for the other simul servers on the machine
(``simul.utils.engine_holds``): each agent's client usually spawns its own
stdio server, and a server that knew nothing of another's hold sent its own
request into the same queue (round-3 audit M3-2). A request to an editor
another server holds is refused with ``InstanceBusy`` before anything is sent,
until that server sees the editor answer again.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..logging import get_logger
from ..utils.engine_holds import SharedHold, SharedHolds, endpoint_key
from ._unreal_errors import INSTANCE_BUSY, UnrealError

logger = get_logger(__name__)

#: Longest the idle probe waits for the editor to accept its connection. A
#: blocked editor's socket still accepts at once, so a connection that is not
#: accepted in this time means nothing is listening to queue behind.
PROBE_CONNECT_TIMEOUT_S: float = 2.0

#: ``details.reason`` of the refusal, shared with the Isaac hold.
HOLD_REASON = "finishing_timed_out_call"


def is_hold_refusal(exc: BaseException) -> bool:
    """Whether ``exc`` refused a request unsent because its editor is held."""
    details = getattr(exc, "details", None)
    return (
        isinstance(exc, UnrealError)
        and exc.error_type == INSTANCE_BUSY
        and isinstance(details, dict)
        and details.get("reason") == HOLD_REASON
    )


@dataclass(frozen=True)
class TimedOutRequest:
    """A request the session stopped waiting for while the editor may still run it.

    Attributes:
        call: The MCP tool that sent it, or ``"<METHOD> <route>"`` outside one.
        route: Its Remote Control route.
        changes_state: Whether it may change the editor (a script or a write
            from a tool that is not read-only), rather than only read.
        since: ``time.monotonic()`` when the session gave up on its answer.
    """

    call: str
    route: str
    changes_state: bool
    since: float


@dataclass
class EditorHold:
    """One editor held after a timed-out request.

    Attributes:
        endpoint: The editor's ``host:port``.
        request: The request that timed out.
        task: The probe that ends the hold once the editor answers.
        record: The hold's record for the other simul servers, or None.
    """

    endpoint: str
    request: TimedOutRequest
    task: "Optional[asyncio.Task[None]]" = None
    record: Optional[Path] = None


def _shared_key(endpoint: str) -> Optional[str]:
    """The ``endpoint_key`` of an editor's ``host:port``, or None when it has no port."""
    host, _, port = endpoint.rpartition(":")
    try:
        return endpoint_key(host, int(port))
    except ValueError:
        return None


class EditorHolds:
    """The editors held after a timeout, by ``host:port``.

    One instance is shared by every session an ``UnrealRuntimeAdapter``
    creates (the MCP server makes a session per call), so a hold outlives the
    call that timed out. A session made on its own has a private one.

    The holds of the other simul servers on the machine count too
    (``elsewhere``): a request to an editor one of them holds is refused the
    same way.

    Args:
        limit: ``unreal.hold_after_timeout``: the longest hold, in seconds;
            0 holds nothing.
        shared: The hold records every simul server on the machine sees;
            ``$SIMUL_HOME/holds`` when None.
    """

    def __init__(self, limit: float, shared: Optional[SharedHolds] = None) -> None:
        self.limit = float(limit)
        self._holds: Dict[str, EditorHold] = {}
        self._shared = shared if shared is not None else SharedHolds("unreal")

    def get(self, endpoint: str) -> Optional[EditorHold]:
        """The hold on ``endpoint``, or None when it is not held."""
        hold = self._holds.get(endpoint)
        if hold is not None and hold.task is not None and hold.task.get_loop().is_closed():
            # Its event loop ended (a test, a CLI run): nothing will end it.
            self._end(hold)
            return None
        return hold

    def elsewhere(self, endpoint: str) -> Optional[SharedHold]:
        """The hold another simul server on the machine keeps on ``endpoint``, or None."""
        key = _shared_key(endpoint)
        return None if key is None else self._shared.foreign(key)

    def hold(self, host: str, port: int, request: TimedOutRequest, headers: Mapping[str, str]) -> bool:
        """Hold the editor at ``host:port`` until it answers a probe sent now.

        The oldest timed-out request stays the one reported: a later one
        queued behind it, and the probe already waits behind both.

        Args:
            host: The editor's host.
            port: Its Remote Control port.
            request: The request that timed out.
            headers: Headers the probe sends (the passphrase).

        Returns:
            Whether the editor is held now (False when ``limit`` is 0).
        """
        if self.limit <= 0:
            return False
        endpoint = f"{host}:{port}"
        if self.get(endpoint) is not None:
            return True
        hold = EditorHold(endpoint=endpoint, request=request)
        task = asyncio.get_running_loop().create_task(self._wait_for_editor(hold, host, port, dict(headers)))
        hold.task = task
        task.add_done_callback(lambda _task: self._end(hold))
        self._holds[endpoint] = hold
        hold.record = self._shared.publish(
            endpoint_key(host, port),
            request.call,
            time.time() - (time.monotonic() - request.since),
            self.limit,
            live=lambda: not task.done() and not task.get_loop().is_closed(),
            details={"timed_out_route": request.route, "changes_state": request.changes_state},
        )
        logger.warning(
            "%s timed out while the editor at %s may still be running it; holding the editor until it "
            "answers again (at most %gs, unreal.hold_after_timeout)",
            request.call,
            endpoint,
            self.limit,
        )
        return True

    def editor_answered(self, endpoint: str, sent_at: float) -> None:
        """End the hold on ``endpoint``: a request sent at ``sent_at`` was answered.

        The editor answers in arrival order, so an answer to a request sent
        after the hold began means it has finished the request that timed out.
        """
        hold = self.get(endpoint)
        if hold is None or sent_at < hold.request.since:
            return
        logger.info(
            "The editor at %s answered, so it has finished %s; released", endpoint, hold.request.call
        )
        self._end(hold)
        if hold.task is not None:
            hold.task.cancel()

    def info(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Describe the hold on ``endpoint``, this server's or another's; None when it is not held.

        Returns:
            ``timed_out_tool``, ``timed_out_route``, ``changes_state``,
            ``busy_for_seconds`` (since the answer timed out) and
            ``hold_limit_seconds``; another server's hold adds ``held_by``
            (its ``pid``).
        """
        hold = self.get(endpoint)
        if hold is None:
            elsewhere = self.elsewhere(endpoint)
            return None if elsewhere is None else elsewhere.info()
        request = hold.request
        return {
            "timed_out_tool": request.call,
            "timed_out_route": request.route,
            "changes_state": request.changes_state,
            "busy_for_seconds": round(time.monotonic() - request.since, 1),
            "hold_limit_seconds": self.limit,
        }

    def refusal(self, endpoint: str) -> Optional[UnrealError]:
        """The InstanceBusy error for a request to a held editor, or None when it is not held."""
        info = self.info(endpoint)
        if info is None:
            return None
        effect = (
            "it may still change the editor's state"
            if info.get("changes_state", True)
            else "it only reads, but the editor serves nothing else until it ends"
        )
        holder = info.get("held_by")
        whose = (
            f" for another simul server on this machine (pid {holder.get('pid')})"
            if isinstance(holder, dict)
            else ""
        )
        ends = "that server sees the editor answer" if whose else "the editor answers"
        return UnrealError(
            f"The editor at {endpoint} is still running {info['timed_out_tool']}{whose}, whose answer timed "
            f"out {info['busy_for_seconds']:.0f}s ago ({effect}). Remote Control runs one request at a "
            "time, so this one would wait behind it, time out too and run after being reported failed: "
            f"nothing was sent. Retry shortly; ping_unreal reports the hold. It ends once {ends} "
            f"again, or after {info['hold_limit_seconds']:g}s (unreal.hold_after_timeout)",
            INSTANCE_BUSY,
            details={"editor": endpoint, "reason": HOLD_REASON, **info},
        )

    def close(self) -> None:
        """Release every hold, with a warning naming its call (the server is stopping)."""
        for hold in list(self._holds.values()):
            logger.warning(
                "Released the Unreal editor at %s while it may still be running %s, which timed out %.0fs ago",
                hold.endpoint,
                hold.request.call,
                time.monotonic() - hold.request.since,
            )
            self._end(hold)
            if hold.task is not None and not hold.task.get_loop().is_closed():
                hold.task.cancel()

    def _end(self, hold: EditorHold) -> None:
        if self._holds.get(hold.endpoint) is hold:
            del self._holds[hold.endpoint]
        record, hold.record = hold.record, None
        self._shared.withdraw(record)

    async def _wait_for_editor(self, hold: EditorHold, host: str, port: int, headers: Dict[str, str]) -> None:
        """Wait, up to the limit, for the editor to answer a probe queued behind the timed-out request."""
        remaining = self.limit - (time.monotonic() - hold.request.since)
        outcome = await editor_answer(host, port, headers, remaining)
        waited = time.monotonic() - hold.request.since
        if outcome == "answered":
            logger.info(
                "The editor at %s finished %s %.1fs after its answer timed out; released",
                hold.endpoint,
                hold.request.call,
                waited,
            )
        elif outcome == "gone":
            logger.info(
                "The editor at %s stopped taking connections %.1fs after %s timed out; released",
                hold.endpoint,
                waited,
                hold.request.call,
            )
        else:
            logger.warning(
                "The editor at %s still had not finished %s %.0fs after its answer timed out "
                "(unreal.hold_after_timeout); releasing it, so new requests may queue behind it",
                hold.endpoint,
                hold.request.call,
                waited,
            )


async def editor_answer(host: str, port: int, headers: Mapping[str, str], timeout: float) -> str:
    """Send ``GET /remote/info`` and wait up to ``timeout`` seconds for the status line.

    Args:
        host: The editor's host.
        port: Its Remote Control port.
        headers: Extra request headers.
        timeout: Longest wait, connecting included.

    Returns:
        ``"answered"`` when the editor answered (any HTTP status), ``"gone"``
        when nothing accepts the connection or it closes without an answer,
        ``"timeout"`` when no answer came in time.
    """
    deadline = time.monotonic() + max(0.0, timeout)
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=max(0.0, min(PROBE_CONNECT_TIMEOUT_S, timeout))
        )
    except (OSError, asyncio.TimeoutError):
        return "gone"
    try:
        lines = ["GET /remote/info HTTP/1.1", f"Host: {host}:{port}", "Connection: close"]
        lines += [f"{name}: {value}" for name, value in headers.items()]
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        await writer.drain()
        status = await asyncio.wait_for(reader.readline(), timeout=max(0.0, deadline - time.monotonic()))
        return "answered" if status.startswith(b"HTTP/") else "gone"
    except asyncio.TimeoutError:
        return "timeout"
    except OSError:
        return "gone"
    finally:
        with contextlib.suppress(Exception):
            writer.close()
