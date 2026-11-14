"""
Isaac Sim bridge-aware client for remote command execution.

Prefers the repo-owned `khemoo.simul` bridge when configured, and can
fall back to the stock Isaac Sim Python socket on port 8226 for compatibility.
Both transports execute inside a running Isaac Sim application.

The stock socket has two flavours that share one reply format:

- Isaac Sim 5.x: `isaacsim.code_editor.vscode` executes on the first packet
  and closes the connection if the client half-closes while an async script
  is still awaiting frames.
- Isaac Sim 6.0+: `isaacsim.code_editor.python_server` buffers until the
  client half-closes (TCP EOF) and only then executes; it also accepts a
  JSON envelope and an optional auth token.

The client probes once per process with a side-effect-free introspection
envelope to learn which flavour it is talking to, then sends EOF only where
the server needs it.

Bridge protocol:
    1. Open TCP connection to bridge host:port
    2. Send a length-prefixed JSON request envelope
    3. Receive a length-prefixed JSON response envelope
    4. Connection closes after each request

Python socket protocol (5.x VS Code ext and 6.0+ python_server ext):
    1. Open TCP connection to host:port
    2. Send the script. 5.x takes raw UTF-8 source; python_server takes the
       JSON envelope {"code", "timeout", "auth_token"}. The timeout only
       bounds the *reply*: python_server answers with a TimeoutError once it
       expires, but the script keeps running inside Kit and its effects may
       still land (v6.0.1 ``python_server/extension.py`` ``_timeout_watchdog``
       only queues a reply). Only the bridge can stop a script.
    3. python_server only: half-close the socket so the server executes
    4. Receive JSON response: {"status":"ok"|"error","output":"..."}
    5. Connection closes after each execution

The 5.x VS Code extension has no framing at all: its ``data_received`` runs
every chunk of bytes that arrives as a separate script, replies to the first
and closes (v5.1.0 ``vscode/extension.py``). A script that arrives in two
reads therefore executes as two fragments, the second possibly with nobody
listening. The client sends that socket at most ``MAX_VSCODE_SCRIPT_BYTES``,
a size one loopback read carries, and refuses larger scripts with a pointer
to the bridge, which reads a length-prefixed frame.

Bridge circuit breaker: a bridge port that is filtered or half-open costs
the full bridge timeout on every call, not the ~1 ms a refused port does.
After ``bridge_failure_threshold`` consecutive transport failures the client
stops dialling the bridge for ``bridge_cooldown_seconds`` and goes straight
to the stock socket; the first call after the cooldown probes the bridge
again, and one success closes the circuit. Readiness probes (``bridge-up``,
``launch``) construct the client with ``bridge_circuit_breaker=False`` so a
bridge that is still binding its port is dialled on every poll.

Fallback is only safe before the bridge has the request: once the
``execute_script`` frame is written the script may already be running in
Kit, and re-sending it to the stock socket would run it twice. Failures
after delivery raise a :class:`BridgeRequestDeliveredError` subclass and are
surfaced to the caller instead of falling back. A reply timeout after
delivery does not count toward the circuit breaker either: the bridge
accepted the request and may still be running it, so routing the next calls
to the stock socket would run new scripts beside the old one. The one reply
that does fall back is a refusal the bridge sends *instead of* running the
script (``UnsafeExecutionDisabled``), since nothing ran.

Authentication: when the client has a token (``ISAAC_SIM__SOCKET_AUTH_TOKEN``
or the file ``simul isaac launch --generate-auth-token`` stores) it sends it
to both transports: in the python_server envelope and in every bridge frame.
A bridge that does not require a token ignores the field.
"""

import asyncio
import json
import logging
import struct
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Dict, Literal, Optional

from .isaac_install import PYTHON_SERVER_EXTENSION, VSCODE_EXTENSION

logger = logging.getLogger(__name__)

SocketProtocol = Literal["auto", "python_server", "vscode"]
SOCKET_PROTOCOLS: tuple[str, ...] = ("auto", "python_server", "vscode")

#: Raw-source header the 6.0 python_server extension reads a token from.
PYTHON_SERVER_TOKEN_HEADER: str = "# isaacsim-python-server-token:"
#: Introspection envelope used to fingerprint the server. On the 5.x VS Code
#: extension it is evaluated as a harmless dict literal; on python_server it
#: is answered without executing anything.
_FLAVOR_PROBE: Dict[str, Any] = {"introspect": "status"}

MAX_RESPONSE_BYTES: int = 10 * 1024 * 1024  # 10 MB cap on response size
MAX_REQUEST_BYTES: int = 1024 * 1024  # 1 MB cap on outgoing bridge payloads
BRIDGE_PROTOCOL_VERSION: str = "1.0"
DEFAULT_BRIDGE_FAILURE_THRESHOLD: int = 3
DEFAULT_BRIDGE_COOLDOWN_SECONDS: float = 30.0
#: Floor for the per-request script timeout sent to the server.
MIN_SCRIPT_TIMEOUT_SECONDS: float = 1.0
#: Extra time the client waits for an ``execute_script`` reply beyond the
#: script budget it sent, so the bridge's own ``ScriptInterrupted`` report
#: arrives before the client gives up on the response.
BRIDGE_SCRIPT_REPLY_MARGIN_SECONDS: float = 5.0
#: Extra time the client waits for a stock-socket reply beyond the script
#: budget, so python_server's own TimeoutError reply arrives first.
SOCKET_SCRIPT_REPLY_MARGIN_SECONDS: float = 1.0
#: Largest payload sent to Isaac Sim 5.x's VS Code socket, in UTF-8 bytes.
#: That extension executes each read separately (see the module docstring), so
#: a script must arrive in one read. Half of loopback's 64 KiB segment, which
#: leaves room for the kernel and asyncio to deliver it whole; links with a
#: smaller MTU (container bridges, SSH tunnels) can still split less, which is
#: why the bridge is the transport to use there.
MAX_VSCODE_SCRIPT_BYTES: int = 32 * 1024
#: Bridge errors sent instead of running the script. Nothing ran, so the
#: client may send the script to the stock socket instead.
BRIDGE_REFUSAL_ERRORS: frozenset = frozenset({"UnsafeExecutionDisabled"})
#: Appended to every report of a script that ran out of time: the Kit-side
#: work a script starts (a stage load, an extension enable, a first render)
#: is not undone by abandoning the script.
TIMEOUT_MAY_STILL_COMPLETE = (
    "Work the script had already started in Kit (a stage load, an extension "
    "enable, a render) may still complete; check the state before retrying."
)


#: Script the idle probe sends to the stock socket. Kit runs scripts one at a
#: time on its main thread, so its reply arrives only once the main thread has
#: finished whatever it was running.
IDLE_PROBE_SCRIPT: str = "print('simul-idle')"
#: Pause between two idle probes that found the bridge still busy.
IDLE_POLL_SECONDS: float = 0.5
#: Bridge actions that leave no work behind when their reply times out: the
#: probes (``ping``, ``get_runtime_info``) and the interrupt itself.
_PROBE_ACTIONS: frozenset = frozenset({"ping", "get_runtime_info", "interrupt"})


class SocketReplyTimeoutError(TimeoutError):
    """The stock socket received the script but did not reply in time.

    Unlike a connect timeout, the script was delivered: Kit may be running it,
    or may run it later, once its main thread is free.
    """


@dataclass(frozen=True)
class UnfinishedRequest:
    """A request this client stopped waiting for while Isaac Sim may still run it.

    Attributes:
        transport: ``"bridge"`` or ``"socket"``, the transport that has it.
        action: The bridge action, or ``"script"`` for the stock socket.
        since: ``time.monotonic()`` when the client gave up on its reply.
    """

    transport: str
    action: str
    since: float


class BridgeCircuitOpenError(ConnectionRefusedError):
    """The bridge was skipped because recent failures opened its circuit.

    A ``ConnectionRefusedError`` so every existing fallback path treats it
    like a bridge that is not there.
    """


class BridgeRequestDeliveredError(Exception):
    """Marker: the bridge failed *after* the request frame was fully written.

    The bridge may already be acting on the request (for ``execute_script``,
    the script may be running in Kit), so callers must not retry it on
    another transport. Concrete subclasses also derive from the matching
    builtin error, so ``except TimeoutError`` / ``except ConnectionError`` /
    ``except ValueError`` handlers still catch them.
    """


class BridgeResponseTimeoutError(BridgeRequestDeliveredError, TimeoutError):
    """The bridge accepted the request but did not reply in time."""


class BridgeResponseConnectionError(BridgeRequestDeliveredError, ConnectionError):
    """The bridge connection broke after the request was delivered."""


class BridgeResponseProtocolError(BridgeRequestDeliveredError, ValueError):
    """The bridge replied after delivery with an unusable response."""


@dataclass(frozen=True)
class ScriptResult:
    """
    Result of executing a Python script inside Isaac Sim.

    Attributes:
        success: Whether execution completed without error.
        output: Captured stdout from the script.
        error_name: Exception class name if execution failed.
        error_value: Exception message if execution failed.
        traceback: Full traceback string if execution failed.
        transport_failure: The client or the transport produced the failure
            (an empty or undecodable reply, a script too large to send, a
            timeout of the wait), rather than the script raising
            ``error_name`` inside Isaac Sim.
    """

    success: bool
    output: str
    error_name: str = ""
    error_value: str = ""
    traceback: str = ""
    transport: str = ""
    transport_failure: bool = False


class IsaacSocketClient:
    """
    Bridge-aware client for communicating with a running Isaac Sim instance.

    The client can use two transports:
    - the repo-owned typed bridge extension
    - the stock Isaac Sim Python socket on ``127.0.0.1:8226``
      (``isaacsim.code_editor.python_server`` on 6.0+,
      ``isaacsim.code_editor.vscode`` on 5.x)

    Thread-safe: an ``asyncio.Lock`` serialises concurrent requests so transport
    interactions never interleave.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8226,
        timeout_seconds: float = 30.0,
        bridge_host: Optional[str] = None,
        bridge_port: Optional[int] = None,
        bridge_socket_path: Optional[str] = None,
        bridge_timeout_seconds: Optional[float] = None,
        prefer_bridge: bool = False,
        fallback_to_vscode: bool = True,
        read_buffer_size: int = 1024 * 1024,
        max_request_bytes: int = MAX_REQUEST_BYTES,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        socket_protocol: SocketProtocol = "auto",
        auth_token: Optional[str] = None,
        script_timeout_seconds: Optional[float] = None,
        bridge_failure_threshold: int = DEFAULT_BRIDGE_FAILURE_THRESHOLD,
        bridge_cooldown_seconds: float = DEFAULT_BRIDGE_COOLDOWN_SECONDS,
        bridge_circuit_breaker: bool = True,
        max_vscode_script_bytes: int = MAX_VSCODE_SCRIPT_BYTES,
    ) -> None:
        """
        Initialize the socket client.

        Args:
            host: Isaac Sim socket server host.
            port: Isaac Sim socket server port.
            timeout_seconds: Timeout for the full send+receive cycle.
            bridge_host: Host for the custom Isaac Sim bridge extension.
            bridge_port: Port for the custom Isaac Sim bridge extension.
            bridge_socket_path: Unix socket to the bridge. Preferred over
                TCP when set — for a containerised Isaac Sim it crosses the
                shared volume directly, with no published port involved.
            bridge_timeout_seconds: Timeout for bridge requests.
            prefer_bridge: Attempt the custom bridge transport before VS Code.
            fallback_to_vscode: Use the VS Code socket when bridge is unavailable.
            read_buffer_size: Max bytes to read per recv call.
            max_request_bytes: Upper bound on bridge request size.
            max_response_bytes: Upper bound on total response size.
            socket_protocol: Which stock socket flavour to speak. "auto"
                probes the server once; "python_server" (Isaac Sim 6.0+)
                half-closes after sending; "vscode" (Isaac Sim 5.x) does not.
            auth_token: Token for a python_server configured with
                ``require_auth``. Sent in the JSON envelope on python_server,
                as a raw-source header on the probe (ignored by 5.x), and in
                every bridge frame, which a bridge requiring auth checks.
            script_timeout_seconds: Budget sent with each script (python_server
                ``timeout`` field and bridge ``execute_script`` payload).
                Defaults to one second less than ``timeout_seconds`` so the
                server reports the overrun before this client's own read
                deadline expires; never below one second. The bridge
                interrupts a script that overruns it; python_server only
                replies with a TimeoutError and lets the script run on; the
                5.x VS Code socket has no such field. Callers pass a longer
                per-call budget for known-heavy operations.
            max_vscode_script_bytes: Largest payload sent to the 5.x VS Code
                socket, which executes each read separately; larger scripts
                are refused with a pointer to the bridge.
            bridge_failure_threshold: Consecutive bridge transport failures
                that open the circuit.
            bridge_cooldown_seconds: How long an open circuit skips the
                bridge before probing it again.
            bridge_circuit_breaker: Set False for readiness probes that poll
                a bridge expected to come up soon; every call then dials the
                bridge regardless of recent failures.
        """
        self._host = host
        self._port = port
        self._timeout_seconds = timeout_seconds
        self._script_timeout_seconds = (
            script_timeout_seconds
            if script_timeout_seconds is not None
            else max(MIN_SCRIPT_TIMEOUT_SECONDS, timeout_seconds - 1.0)
        )
        if bridge_failure_threshold < 1:
            raise ValueError("bridge_failure_threshold must be at least 1")
        if bridge_cooldown_seconds < 0:
            raise ValueError("bridge_cooldown_seconds must not be negative")
        self._bridge_failure_threshold = bridge_failure_threshold
        self._bridge_cooldown_seconds = bridge_cooldown_seconds
        self._bridge_circuit_breaker = bridge_circuit_breaker
        self._bridge_consecutive_failures = 0
        self._bridge_circuit_opened_at: Optional[float] = None
        self._bridge_host = bridge_host
        self._bridge_port = bridge_port
        self._bridge_socket_path = bridge_socket_path
        self._bridge_timeout_seconds = (
            bridge_timeout_seconds
            if bridge_timeout_seconds is not None
            else timeout_seconds
        )
        self._prefer_bridge = prefer_bridge and (
            bridge_socket_path is not None
            or (bridge_host is not None and bridge_port is not None)
        )
        self._fallback_to_vscode = fallback_to_vscode
        self._read_buffer_size = read_buffer_size
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        if socket_protocol not in SOCKET_PROTOCOLS:
            raise ValueError(
                f"socket_protocol must be one of {SOCKET_PROTOCOLS}, got {socket_protocol!r}"
            )
        self._socket_protocol: SocketProtocol = socket_protocol
        self._auth_token = auth_token or None
        self._max_vscode_script_bytes = max_vscode_script_bytes
        # Resolved flavour once "auto" has probed; reset when the server goes away
        # so an Isaac Sim restarted on a different version is re-detected.
        self._detected_socket_protocol: Optional[str] = (
            None if socket_protocol == "auto" else socket_protocol
        )
        self.__lock: Optional[asyncio.Lock] = None
        # The oldest request whose reply this client gave up on while the
        # engine may still be running it; wait_until_idle clears it.
        self._unfinished: Optional[UnfinishedRequest] = None
        # Tracks "{bridge}->{vscode}" pairs that have already produced a
        # WARNING-level fallback log so subsequent failures stay at DEBUG.
        self._logged_bridge_failures: set[str] = set()

    @property
    def _lock(self) -> asyncio.Lock:
        """Lazy-init the asyncio.Lock for Python <3.10 safety."""
        if self.__lock is None:
            self.__lock = asyncio.Lock()
        return self.__lock

    @property
    def unfinished_request(self) -> Optional[UnfinishedRequest]:
        """The request Isaac Sim may still be running after its reply timed out, if any.

        Set when a script or bridge action was delivered and its reply did not
        arrive in time: a stock-socket read timeout, python_server's
        ``TimeoutError`` reply, a bridge reply timeout or ``RequestTimeout``.
        Kit goes on running the request and runs a later one only after it,
        so work sent meanwhile times out too and runs after it was reported
        failed. ``wait_until_idle`` clears it once the engine has finished.
        """
        return self._unfinished

    def _note_unfinished(self, transport: str, action: str) -> None:
        """Remember a delivered request whose reply timed out (the oldest one wins)."""
        if self._unfinished is None:
            self._unfinished = UnfinishedRequest(transport, action, time.monotonic())

    def forget_unfinished_request(self) -> None:
        """Stop tracking the unfinished request, after giving up on waiting for it."""
        self._unfinished = None

    async def wait_until_idle(self, timeout: float) -> bool:
        """Wait until Isaac Sim has finished the request whose reply timed out.

        The probe queues behind that request. On the stock socket it is a
        trivial script, which Kit runs only once its main thread is free. On
        the bridge it is ``get_runtime_info``, which answers once the main
        thread is free and reports whether a script or action is still
        running (a coroutine script awaiting frames leaves the main thread
        free); it is asked again every ``IDLE_POLL_SECONDS`` while busy.
        Nothing listening any more (Isaac Sim exited or restarted) counts as
        idle: nothing runs the request then.

        On the stock sockets a coroutine script that awaits frames leaves the
        main thread free between frames, so the probe can answer while that
        script still runs; only the bridge reports it.

        Args:
            timeout: Longest wait, in seconds.

        Returns:
            True once the engine is idle (``unfinished_request`` is then
            cleared), False when ``timeout`` passed first.
        """
        unfinished = self._unfinished
        if unfinished is None:
            return True
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                idle = await asyncio.wait_for(
                    self._engine_is_idle(unfinished, remaining), timeout=remaining
                )
            except asyncio.TimeoutError:
                return False
            if idle:
                if self._unfinished is unfinished:
                    self._unfinished = None
                return True
            pause = min(IDLE_POLL_SECONDS, deadline - time.monotonic())
            if pause > 0:
                await asyncio.sleep(pause)

    async def _engine_is_idle(self, unfinished: UnfinishedRequest, remaining: float) -> bool:
        """Probe once, waiting up to ``remaining`` seconds; True when the engine is free."""
        try:
            if unfinished.transport == "bridge":
                response = await self._dial_bridge(
                    "get_runtime_info", {"include_prim_count": False}, read_timeout=remaining
                )
                payload = response.get("payload")
                activity = payload.get("bridge") if isinstance(payload, dict) else None
                # An answer means the main thread is free; a bridge without
                # the activity report cannot say more than that.
                return not (isinstance(activity, dict) and activity.get("busy"))
            result = await self._execute_vscode_unlocked(
                IDLE_PROBE_SCRIPT,
                max(MIN_SCRIPT_TIMEOUT_SECONDS, remaining),
                read_timeout=remaining,
                note_unfinished=False,
            )
            # python_server's watchdog answers TimeoutError while the probe
            # still waits behind the main thread; any other reply means Kit
            # got to the probe.
            return result.error_name != "TimeoutError"
        except TimeoutError:
            return False
        except (ConnectionError, OSError):
            # Refused or reset: Isaac Sim is gone, and the request with it.
            return True
        except ValueError:
            # Something answered, if not in the protocol: the main thread is free.
            return True

    @property
    def _bridge_configured(self) -> bool:
        """Return True when a bridge transport (socket or host:port) is set."""
        return self._bridge_socket_path is not None or (
            self._bridge_host is not None and self._bridge_port is not None
        )

    @property
    def address(self) -> str:
        """Return the target address as host:port string."""
        if self._prefer_bridge and self._bridge_configured:
            return self.bridge_endpoint
        return f"{self._host}:{self._port}"

    @property
    def bridge_endpoint(self) -> str:
        """The bridge endpoint actually dialled: socket path or host:port.

        Error messages and log-dedup keys must name this, not the TCP pair —
        a socket-only client reporting "127.0.0.1:8229" (or "None:None")
        points the operator at a port that was never dialled.
        """
        if self._bridge_socket_path is not None:
            return self._bridge_socket_path
        return f"{self._bridge_host}:{self._bridge_port}"

    @property
    def bridge_address(self) -> str:
        """Return the bridge address as host:port string."""
        if not self._bridge_configured:
            return ""
        return f"{self._bridge_host}:{self._bridge_port}"

    @property
    def vscode_address(self) -> str:
        """Return the VS Code socket address as host:port string."""
        return f"{self._host}:{self._port}"

    @property
    def timeout_seconds(self) -> float:
        """Return the configured timeout in seconds."""
        return self._timeout_seconds

    @property
    def bridge_enabled(self) -> bool:
        """Return True when bridge transport is configured and preferred."""
        return self._prefer_bridge

    @property
    def fallback_to_vscode(self) -> bool:
        """Return True when VS Code fallback is enabled."""
        return self._fallback_to_vscode

    @property
    def socket_protocol(self) -> str:
        """Return the stock socket flavour in use: python_server, vscode, or auto."""
        return self._detected_socket_protocol or self._socket_protocol

    @property
    def script_timeout_seconds(self) -> float:
        """Return the per-script execution budget asked of the server."""
        return self._script_timeout_seconds

    @property
    def bridge_circuit_open(self) -> bool:
        """Whether bridge calls are currently being skipped after repeated failures.

        Turns back to False once the cooldown has elapsed: the next call is
        allowed through as a probe, and its outcome decides whether the
        circuit closes or reopens.
        """
        if self._bridge_circuit_opened_at is None:
            return False
        elapsed = time.monotonic() - self._bridge_circuit_opened_at
        return elapsed < self._bridge_cooldown_seconds

    @property
    def bridge_consecutive_failures(self) -> int:
        """Return how many bridge transport calls in a row have failed."""
        return self._bridge_consecutive_failures

    def _resolve_script_timeout(self, timeout_seconds: Optional[float]) -> float:
        """Return the script budget for one call: its override, else the default."""
        if timeout_seconds is None:
            return self._script_timeout_seconds
        return max(MIN_SCRIPT_TIMEOUT_SECONDS, float(timeout_seconds))

    async def execute(
        self, code: str, *, timeout_seconds: Optional[float] = None
    ) -> ScriptResult:
        """
        Execute Python code inside the running Isaac Sim process.

        When bridge transport is preferred, raw script execution is attempted
        on the bridge first. If the bridge cannot be reached (refused, connect
        timeout, open circuit, or the request frame could not be written) and
        fallback is enabled, the client retries through the VS Code socket.
        A failure after the frame was delivered is raised instead: the script
        may already be running in Kit and must not be sent a second time. A
        bridge that refuses raw scripts (``allow_unsafe_execution=false``)
        says so without running anything, so that refusal falls back too.

        Args:
            code: Python source code to execute in Isaac Sim's Python scope.
            timeout_seconds: Script budget for this call, for operations known
                to outlast the default (opening a stage, a first render).
                Defaults to ``script_timeout_seconds``.

        Returns:
            ScriptResult with stdout capture and error details if any.

        Raises:
            ConnectionRefusedError: If Isaac Sim is not running or the
                extension is not enabled.
            TimeoutError: If the connection or execution exceeds timeout_seconds.
            BridgeRequestDeliveredError: The bridge received the script but
                the reply failed; never retried on the stock socket.
        """
        script_timeout = self._resolve_script_timeout(timeout_seconds)
        async with self._lock:
            if self._prefer_bridge:
                try:
                    result = await self._execute_bridge_script(code, script_timeout)
                except BridgeRequestDeliveredError:
                    raise
                except (
                    ConnectionRefusedError,
                    ConnectionError,
                    OSError,
                    TimeoutError,
                    ValueError,
                ) as exc:
                    self._log_bridge_fallback(f"Bridge transport failed: {exc}")
                    if not self._fallback_to_vscode:
                        raise
                else:
                    if not (
                        self._fallback_to_vscode
                        and result.error_name in BRIDGE_REFUSAL_ERRORS
                    ):
                        return result
                    self._log_bridge_fallback(
                        f"Bridge refused the script without running it ({result.error_name})"
                    )
            return await self._execute_vscode_unlocked(code, script_timeout)

    def _log_bridge_fallback(self, reason: str) -> None:
        """Log a fall back to the stock socket, at WARNING once per destination pair.

        Subsequent fallbacks with the same destinations drop to DEBUG so a
        long-lived process running without the bridge does not spam the log
        on every tool call.
        """
        fallback_key = f"{self.bridge_endpoint}->{self.vscode_address}"
        level = (
            logging.WARNING
            if fallback_key not in self._logged_bridge_failures
            else logging.DEBUG
        )
        self._logged_bridge_failures.add(fallback_key)
        logger.log(
            level,
            "%s; falling back from the bridge at %s to the VS Code socket %s",
            reason,
            self.bridge_endpoint,
            self.vscode_address,
        )

    async def execute_bridge_script_only(
        self, code: str, *, timeout_seconds: Optional[float] = None
    ) -> ScriptResult:
        """Execute raw Python only through the bridge transport."""
        script_timeout = self._resolve_script_timeout(timeout_seconds)
        async with self._lock:
            return await self._execute_bridge_script(code, script_timeout)

    async def execute_vscode_only(
        self,
        code: str,
        *,
        timeout_seconds: Optional[float] = None,
        note_unfinished: bool = True,
    ) -> ScriptResult:
        """Execute raw Python only through the VS Code socket transport.

        ``note_unfinished=False`` keeps a reply timeout out of
        ``unfinished_request``, for probes that run outside an instance's
        lock (``list_isaac_instances``).
        """
        script_timeout = self._resolve_script_timeout(timeout_seconds)
        async with self._lock:
            return await self._execute_vscode_unlocked(
                code, script_timeout, note_unfinished=note_unfinished
            )

    async def _execute_vscode_unlocked(
        self,
        code: str,
        script_timeout: Optional[float] = None,
        *,
        read_timeout: Optional[float] = None,
        note_unfinished: bool = True,
    ) -> ScriptResult:
        """
        Execute Python code through the stock Isaac Sim Python socket.

        Args:
            code: Python source code to execute.
            script_timeout: Budget for this script; the default when None.
            read_timeout: Deadline for the reply; by default the larger of the
                client timeout and the budget plus a margin.
            note_unfinished: Record a delivered script whose reply timed out
                as ``unfinished_request`` (off for pings and idle probes).

        Returns:
            ScriptResult with execution output.

        Raises:
            ConnectionRefusedError: Cannot reach Isaac Sim.
            TimeoutError: Connection or read timed out.
        """
        budget = self._script_timeout_seconds if script_timeout is None else script_timeout
        if self._detected_socket_protocol is None:
            self._detected_socket_protocol = await self._detect_socket_protocol()
        is_python_server = self._detected_socket_protocol == "python_server"
        if is_python_server:
            # The envelope carries the per-request timeout, which makes
            # python_server reply in time; it does not stop the script. A
            # stale flavour is harmless: the 5.x extension evaluates the
            # envelope as a dict literal and then closes on EOF, which
            # re-probes below.
            envelope: Dict[str, Any] = {
                "code": code,
                "timeout": budget,
            }
            if self._auth_token:
                envelope["auth_token"] = self._auth_token
            payload = json.dumps(envelope)
        else:
            payload = code
            if self._auth_token:
                payload = f"{PYTHON_SERVER_TOKEN_HEADER} {self._auth_token}\n{code}"
        payload_bytes = payload.encode("utf-8")
        if not is_python_server and len(payload_bytes) > self._max_vscode_script_bytes:
            # Refused before anything is sent: past one read, the 5.x
            # extension would execute the script as separate fragments.
            return ScriptResult(
                success=False,
                output="",
                error_name="ScriptTooLargeForSocket",
                transport_failure=True,
                error_value=(
                    f"Script is {len(payload_bytes)} bytes; Isaac Sim 5.x's "
                    f"{VSCODE_EXTENSION} socket at {self.vscode_address} executes each "
                    "read it receives as a separate script, so simul sends it at most "
                    f"{self._max_vscode_script_bytes} bytes. Enable the khemoo.simul "
                    "bridge (`simul isaac bridge-up`), which reads the whole script "
                    "before running it, or send a smaller script."
                ),
                transport="vscode",
            )
        try:
            raw_response = await self._socket_round_trip(
                payload_bytes,
                half_close=is_python_server,
                read_timeout=(
                    read_timeout
                    if read_timeout is not None
                    else max(self._timeout_seconds, budget + SOCKET_SCRIPT_REPLY_MARGIN_SECONDS)
                ),
            )
        except SocketReplyTimeoutError:
            if note_unfinished:
                self._note_unfinished("socket", "script")
            self._forget_detected_protocol()
            raise
        except (ConnectionRefusedError, TimeoutError, OSError, ValueError):
            self._forget_detected_protocol()
            raise
        result = self._parse_response(raw_response)
        if result.error_name == "EmptyResponse":
            # A 5.x server that received EOF closes without replying. That is
            # what a stale "python_server" flavour looks like, so re-probe.
            self._forget_detected_protocol()
        elif is_python_server and result.error_name == "TimeoutError":
            # python_server's watchdog only sends this reply; the script runs on.
            if note_unfinished:
                self._note_unfinished("socket", "script")
            result = replace(
                result,
                error_value=(
                    f"{result.error_value}. Isaac Sim 6.0's python_server stops "
                    "waiting but does not stop the script, which keeps running "
                    f"inside Kit. {TIMEOUT_MAY_STILL_COMPLETE}"
                ),
                transport_failure=True,
            )
        return result

    def _forget_detected_protocol(self) -> None:
        """Drop a probed flavour so the next call detects it again.

        Isaac Sim restarted on the other major version does not announce
        itself: a wrong cached flavour surfaces as a timeout (6.0 buffers
        forever without EOF) or an empty reply (5.x closes on EOF), never as a
        connection error. Without this the first mismatch would repeat until
        the process restarts.
        """
        if self._socket_protocol == "auto":
            self._detected_socket_protocol = None

    async def _detect_socket_protocol(self) -> str:
        """Fingerprint the stock socket server with one side-effect-free request.

        Returns:
            "python_server" when the reply carries the introspection status
            payload, otherwise "vscode".
        """
        probe: Dict[str, Any] = dict(_FLAVOR_PROBE)
        if self._auth_token:
            probe["auth_token"] = self._auth_token
        raw = await self._socket_round_trip(
            json.dumps(probe).encode("utf-8"), half_close=True
        )
        flavor = "vscode"
        try:
            reply = json.loads(raw)
        except json.JSONDecodeError:
            reply = None
        if isinstance(reply, dict):
            result = reply.get("result")
            if isinstance(result, dict) and "uptime_seconds" in result:
                flavor = "python_server"
            elif reply.get("ename") == "AuthenticationError":
                # Only python_server knows this error; the token is wrong or
                # missing, which the real request will report verbatim.
                flavor = "python_server"
        logger.info(
            "Isaac Sim socket at %s speaks the %s protocol", self.vscode_address, flavor
        )
        return flavor

    async def _socket_round_trip(
        self,
        payload: bytes,
        *,
        half_close: bool,
        read_timeout: Optional[float] = None,
    ) -> str:
        """Send one request on the stock socket and read the reply until close.

        Args:
            payload: Bytes to send.
            half_close: Send TCP EOF after the payload. Required by the 6.0
                python_server, which buffers until EOF; fatal on the 5.x VS Code
                extension for scripts that await across frames.
            read_timeout: Deadline for the whole reply; ``timeout_seconds``
                when None.

        Returns:
            The raw UTF-8 reply.
        """
        read_deadline_seconds = (
            self._timeout_seconds if read_timeout is None else read_timeout
        )
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Connection to Isaac Sim at {self.vscode_address} timed out "
                f"after {self._timeout_seconds}s."
            )
        except (ConnectionRefusedError, OSError) as exc:
            raise ConnectionRefusedError(
                f"Cannot connect to Isaac Sim at {self.vscode_address}. "
                "Ensure Isaac Sim is running with its Python socket server enabled "
                f"({PYTHON_SERVER_EXTENSION} on 6.0+, {VSCODE_EXTENSION} on 5.x)."
            ) from exc

        try:
            writer.write(payload)
            await writer.drain()
            if half_close and writer.can_write_eof():
                writer.write_eof()

            chunks: list[bytes] = []
            total_bytes: int = 0
            deadline: float = time.monotonic() + read_deadline_seconds
            while True:
                remaining: float = deadline - time.monotonic()
                if remaining <= 0:
                    raise SocketReplyTimeoutError(
                        f"Overall read from Isaac Sim at {self.vscode_address} timed out "
                        f"after {read_deadline_seconds}s. The stock socket cannot stop "
                        "a script, so it may still be running inside Isaac Sim. "
                        f"{TIMEOUT_MAY_STILL_COMPLETE}"
                    )
                try:
                    chunk = await asyncio.wait_for(
                        reader.read(self._read_buffer_size),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    raise SocketReplyTimeoutError(
                        f"Overall read from Isaac Sim at {self.vscode_address} timed out "
                        f"after {read_deadline_seconds}s. The stock socket cannot stop "
                        "a script, so it may still be running inside Isaac Sim. "
                        f"{TIMEOUT_MAY_STILL_COMPLETE}"
                    ) from None
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > self._max_response_bytes:
                    raise ValueError(
                        f"Response from Isaac Sim exceeded {self._max_response_bytes} bytes limit."
                    )
                chunks.append(chunk)

            return b"".join(chunks).decode("utf-8")

        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError) as exc:
                logger.warning("Failed to close socket cleanly: %s", exc)

    async def _execute_bridge_script(
        self, code: str, script_timeout: Optional[float] = None
    ) -> ScriptResult:
        """Execute Python code through the custom bridge extension.

        The bridge enforces ``timeout`` itself: a script still running when it
        expires is interrupted inside Kit, so the failure comes back as a
        ``ScriptInterrupted`` error rather than a client-side read timeout
        with the script still wedging the editor. Interrupting the script
        does not undo Kit work it already started, which the error says.
        """
        budget = self._script_timeout_seconds if script_timeout is None else script_timeout
        response = await self._bridge_request(
            "execute_script",
            {"code": code, "timeout": budget},
            read_timeout=max(
                self._bridge_timeout_seconds,
                budget + BRIDGE_SCRIPT_REPLY_MARGIN_SECONDS,
            ),
        )
        if response.get("status") == "ok":
            payload = response.get("payload", {})
            return ScriptResult(
                success=True,
                output=str(payload.get("output", "")),
                transport="bridge",
            )

        error = response.get("error", {})
        error_name = str(error.get("name", "BridgeError"))
        error_value = str(error.get("message", "Bridge request failed"))
        timed_out = error_name == "ScriptInterrupted" and error_value.startswith("timed out")
        if timed_out:
            error_value = (
                f"Script {error_value} and was interrupted by the bridge. "
                f"{TIMEOUT_MAY_STILL_COMPLETE}"
            )
        return ScriptResult(
            success=False,
            output=str(response.get("payload", {}).get("output", "")),
            error_name=error_name,
            error_value=error_value,
            traceback=str(error.get("traceback", "")),
            transport="bridge",
            transport_failure=timed_out,
        )

    async def ping(self) -> bool:
        """
        Check if Isaac Sim is reachable by executing a trivial script.

        Takes no request lock: each probe is its own connection, so a ping
        does not wait behind a call already in flight on this client (it used
        to queue behind a long script for as long as the script ran). Whether
        the engine answers while it runs a script is up to the engine: the
        bridge does between frames; a synchronous script blocks every
        transport until it returns.

        Returns:
            True if Isaac Sim responded successfully, False otherwise,
            including when what answered sent a reply that is not the
            protocol's (undecodable bytes, a malformed frame).
        """
        if self._prefer_bridge:
            try:
                response = await self._bridge_request("ping", {})
                if response.get("status") == "ok":
                    payload = response.get("payload", {})
                    if not isinstance(payload, dict):
                        return False
                    return bool(payload.get("reachable", True))
            except (ConnectionRefusedError, TimeoutError, OSError, ValueError):
                if not self._fallback_to_vscode:
                    return False
        try:
            result = await self._execute_vscode_unlocked("print('pong')", note_unfinished=False)
            return result.success and "pong" in result.output
        except (ConnectionRefusedError, TimeoutError, OSError, ValueError):
            # ValueError covers UnicodeDecodeError and an oversized reply.
            return False

    @property
    def busy(self) -> bool:
        """Whether a request (other than a ping) is in flight on this client."""
        return self._lock.locked()

    async def bridge_request(
        self,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        read_timeout: Optional[float] = None,
        note_unfinished: bool = True,
    ) -> Dict[str, Any]:
        """Send a typed request to the bridge transport without fallback.

        Args:
            action: Bridge action name.
            payload: Action payload.
            read_timeout: How long to wait for the reply once the request is
                sent, for actions whose duration scales with their input
                (``simulation_control`` step); the bridge timeout when None.
            note_unfinished: Record a reply timeout as ``unfinished_request``
                (off for probes outside an instance's lock).
        """
        async with self._lock:
            return await self._bridge_request(
                action,
                payload or {},
                read_timeout=read_timeout,
                note_unfinished=note_unfinished,
            )

    async def interrupt_bridge_script(self) -> Dict[str, Any]:
        """Ask the bridge to stop the script it is running.

        Deliberately skips this client's request lock and the circuit
        breaker: the script being interrupted is usually the call holding
        the lock, and an operator asking for a stop wants the wire tried
        regardless of recent failures.

        Returns:
            The bridge response envelope for the ``interrupt`` action.
        """
        return await self._bridge_request("interrupt", {}, ignore_circuit=True)

    async def _bridge_request(
        self,
        action: str,
        payload: Dict[str, Any],
        *,
        ignore_circuit: bool = False,
        read_timeout: Optional[float] = None,
        note_unfinished: bool = True,
    ) -> Dict[str, Any]:
        """Send a typed request to the custom bridge extension.

        Args:
            action: Bridge action name.
            payload: Action payload.
            ignore_circuit: Dial the bridge even while its circuit is open.
            read_timeout: How long to wait for the reply once the request is
                sent; defaults to the bridge timeout.
            note_unfinished: Record a reply timeout as ``unfinished_request``
                (probe actions never are).

        Raises:
            BridgeCircuitOpenError: The circuit is open and ``ignore_circuit``
                is False; nothing was dialled.
        """
        if not self._bridge_configured:
            raise ConnectionRefusedError("Bridge transport is not configured.")
        if not ignore_circuit and self._bridge_circuit_breaker and self.bridge_circuit_open:
            remaining = self._bridge_cooldown_seconds - (
                time.monotonic() - (self._bridge_circuit_opened_at or 0.0)
            )
            raise BridgeCircuitOpenError(
                f"Isaac bridge at {self.bridge_endpoint} skipped: "
                f"{self._bridge_consecutive_failures} consecutive failures opened "
                f"its circuit; retrying in {max(0.0, remaining):.0f}s."
            )
        try:
            response = await self._dial_bridge(action, payload, read_timeout=read_timeout)
        except BridgeResponseTimeoutError:
            # The bridge accepted the connection and holds the request; it is
            # slow, not unreachable. Counting this would open the circuit
            # after a few slow calls and send new scripts to the stock socket
            # while the bridge still runs the old one. A connection the peer
            # drops without replying (a port proxy with nothing behind it)
            # still counts below.
            if note_unfinished and action not in _PROBE_ACTIONS:
                self._note_unfinished("bridge", action)
            raise
        except (ConnectionRefusedError, ConnectionError, TimeoutError, OSError):
            self._record_bridge_failure()
            raise
        self._record_bridge_success()
        error = response.get("error") if response.get("status") != "ok" else None
        if (
            isinstance(error, dict)
            and error.get("name") == "RequestTimeout"
            and note_unfinished
            and action not in _PROBE_ACTIONS
        ):
            # The bridge stopped waiting for its handler, which runs on.
            self._note_unfinished("bridge", action)
        return response

    def _record_bridge_failure(self) -> None:
        """Count a transport failure and open the circuit at the threshold."""
        self._bridge_consecutive_failures += 1
        if (
            not self._bridge_circuit_breaker
            or self._bridge_consecutive_failures < self._bridge_failure_threshold
        ):
            return
        was_open = self._bridge_circuit_opened_at is not None
        self._bridge_circuit_opened_at = time.monotonic()
        if not was_open:
            logger.warning(
                "Isaac bridge at %s failed %d times in a row; skipping it for %.0fs "
                "and using the stock socket at %s directly.",
                self.bridge_endpoint,
                self._bridge_consecutive_failures,
                self._bridge_cooldown_seconds,
                self.vscode_address,
            )

    def _record_bridge_success(self) -> None:
        """Close the circuit after a bridge call that got a response."""
        if self._bridge_circuit_opened_at is not None:
            logger.info(
                "Isaac bridge at %s is answering again; circuit closed.",
                self.bridge_endpoint,
            )
        self._bridge_consecutive_failures = 0
        self._bridge_circuit_opened_at = None

    async def _dial_bridge(
        self,
        action: str,
        payload: Dict[str, Any],
        *,
        read_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform one bridge round trip: connect, send the frame, read the reply.

        Errors before the frame is fully written (refused, connect timeout,
        write failure) propagate as the builtin exception. Errors after it are
        re-raised as :class:`BridgeRequestDeliveredError` subclasses so callers
        know the bridge may already be acting on the request.
        """
        reply_timeout = (
            read_timeout if read_timeout is not None else self._bridge_timeout_seconds
        )
        request: Dict[str, Any] = {
            "protocol_version": BRIDGE_PROTOCOL_VERSION,
            "request_id": str(uuid.uuid4()),
            "action": action,
            "payload": payload,
        }
        if self._auth_token:
            # Beside the payload, never in it: no action handler sees it, and
            # a bridge without authentication ignores the field.
            request["auth_token"] = self._auth_token
        request_bytes = json.dumps(request, separators=(",", ":")).encode("utf-8")
        if len(request_bytes) > self._max_request_bytes:
            raise ValueError(
                f"Bridge request exceeded {self._max_request_bytes} bytes limit."
            )
        frame = struct.pack(">I", len(request_bytes)) + request_bytes

        endpoint = self.bridge_endpoint
        if self._bridge_socket_path is not None:
            connect = asyncio.open_unix_connection(self._bridge_socket_path)
        else:
            connect = asyncio.open_connection(self._bridge_host, self._bridge_port)
        try:
            reader, writer = await asyncio.wait_for(
                connect, timeout=self._bridge_timeout_seconds
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Connection to Isaac bridge at {endpoint} timed out after "
                f"{self._bridge_timeout_seconds}s."
            )
        except (ConnectionRefusedError, OSError) as exc:
            raise ConnectionRefusedError(
                f"Cannot connect to Isaac bridge at {endpoint}."
            ) from exc

        try:
            writer.write(frame)
            await writer.drain()
            # From here on the bridge holds the whole request and may be
            # acting on it; nothing below may look like "bridge unreachable".
            # The bridge's raw-code path reads until EOF or a 5 s idle timeout;
            # without this it would wait out that timeout on every request that
            # reached it.
            try:
                if writer.can_write_eof():
                    writer.write_eof()
            except (OSError, RuntimeError):
                pass

            try:
                header = await asyncio.wait_for(
                    reader.readexactly(4),
                    timeout=reply_timeout,
                )
                payload_size = struct.unpack(">I", header)[0]
                if payload_size > self._max_response_bytes:
                    raise BridgeResponseProtocolError(
                        f"Bridge response exceeded {self._max_response_bytes} bytes limit."
                    )
                response_bytes = await asyncio.wait_for(
                    reader.readexactly(payload_size),
                    timeout=reply_timeout,
                )
            except asyncio.IncompleteReadError as exc:
                raise BridgeResponseConnectionError(
                    f"Bridge at {endpoint} closed connection before sending full response "
                    f"to {action!r}; the request was delivered and was not retried."
                ) from exc
            except asyncio.TimeoutError:
                raise BridgeResponseTimeoutError(
                    f"No response from Isaac bridge at {endpoint} within "
                    f"{reply_timeout}s of sending {action!r}; the request was "
                    "delivered and may still be running, so it was not retried."
                ) from None
            except BridgeRequestDeliveredError:
                raise
            except OSError as exc:
                raise BridgeResponseConnectionError(
                    f"Connection to Isaac bridge at {endpoint} failed after sending "
                    f"{action!r}: {exc}; the request was not retried."
                ) from exc

            try:
                response: Dict[str, Any] = json.loads(response_bytes.decode("utf-8"))
            except ValueError as exc:
                raise BridgeResponseProtocolError(
                    f"Bridge at {endpoint} sent an undecodable response to {action!r}: {exc}"
                ) from exc
            if not isinstance(response, dict):
                raise BridgeResponseProtocolError("Bridge response must be a JSON object.")

            expected_id: str = request["request_id"]
            if response.get("request_id") != expected_id:
                logger.warning(
                    "Bridge response request_id mismatch: expected %s, got %s",
                    expected_id,
                    response.get("request_id"),
                )

            return response
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError) as exc:
                logger.debug("Failed to close bridge socket cleanly: %s", exc)

    @staticmethod
    def _parse_response(raw: str) -> ScriptResult:
        """
        Parse the JSON response from the VS Code extension socket server.

        Args:
            raw: Raw JSON string from the server.

        Returns:
            Parsed ScriptResult.
        """
        if not raw.strip():
            return ScriptResult(
                success=False,
                output="",
                error_name="EmptyResponse",
                error_value="No response from Isaac Sim",
                transport="vscode",
                transport_failure=True,
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ScriptResult(
                success=False,
                output=raw,
                error_name="JSONDecodeError",
                error_value=str(exc),
                transport="vscode",
                transport_failure=True,
            )

        status = data.get("status", "error")
        output = data.get("output", "")

        if status == "ok":
            return ScriptResult(success=True, output=output, transport="vscode")

        traceback_lines = data.get("traceback", [])
        traceback_str = (
            "\n".join(traceback_lines)
            if isinstance(traceback_lines, list)
            else str(traceback_lines)
        )

        return ScriptResult(
            success=False,
            output=output,
            error_name=data.get("ename", "UnknownError"),
            error_value=data.get("evalue", ""),
            traceback=traceback_str,
            transport="vscode",
        )
