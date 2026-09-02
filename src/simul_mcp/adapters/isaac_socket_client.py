"""
Isaac Sim bridge-aware client for remote command execution.

Prefers the repo-owned `khemoo.simul.mcp` bridge when configured, and can
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
    2. Send Python code as UTF-8 bytes (optionally prefixed by a token header)
    3. python_server only: half-close the socket so the server executes
    4. Receive JSON response: {"status":"ok"|"error","output":"..."}
    5. Connection closes after each execution
"""

import asyncio
import json
import logging
import struct
import time
import uuid
from dataclasses import dataclass
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
    """

    success: bool
    output: str
    error_name: str = ""
    error_value: str = ""
    traceback: str = ""
    transport: str = ""


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
                ``require_auth``. Sent as a raw-source header; ignored by 5.x.
        """
        self._host = host
        self._port = port
        self._timeout_seconds = timeout_seconds
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
        # Resolved flavour once "auto" has probed; reset when the server goes away
        # so an Isaac Sim restarted on a different version is re-detected.
        self._detected_socket_protocol: Optional[str] = (
            None if socket_protocol == "auto" else socket_protocol
        )
        self.__lock: Optional[asyncio.Lock] = None
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

    async def execute(self, code: str) -> ScriptResult:
        """
        Execute Python code inside the running Isaac Sim process.

        When bridge transport is preferred, raw script execution is attempted
        on the bridge first. If bridge raw-script execution fails and fallback
        is enabled, the client retries through the VS Code socket.

        Args:
            code: Python source code to execute in Isaac Sim's Python scope.

        Returns:
            ScriptResult with stdout capture and error details if any.

        Raises:
            ConnectionRefusedError: If Isaac Sim is not running or the
                extension is not enabled.
            TimeoutError: If the connection or execution exceeds timeout_seconds.
        """
        async with self._lock:
            if self._prefer_bridge:
                try:
                    return await self._execute_bridge_script(code)
                except (
                    ConnectionRefusedError,
                    ConnectionError,
                    OSError,
                    TimeoutError,
                    ValueError,
                ) as exc:
                    # Log the first failure per (bridge, vscode) pair at WARNING.
                    # Subsequent failures with the same destinations drop to DEBUG
                    # so a long-lived process running without the bridge does not
                    # spam the log on every tool call.
                    fallback_key = f"{self.bridge_endpoint}->{self.vscode_address}"
                    level = (
                        logging.WARNING
                        if fallback_key not in self._logged_bridge_failures
                        else logging.DEBUG
                    )
                    self._logged_bridge_failures.add(fallback_key)
                    logger.log(
                        level,
                        "Bridge transport failed at %s, falling back to VS Code socket %s: %s",
                        self.bridge_endpoint,
                        self.vscode_address,
                        exc,
                    )
                    if not self._fallback_to_vscode:
                        raise
            return await self._execute_vscode_unlocked(code)

    async def execute_bridge_script_only(self, code: str) -> ScriptResult:
        """Execute raw Python only through the bridge transport."""
        async with self._lock:
            return await self._execute_bridge_script(code)

    async def execute_vscode_only(self, code: str) -> ScriptResult:
        """Execute raw Python only through the VS Code socket transport."""
        async with self._lock:
            return await self._execute_vscode_unlocked(code)

    async def _execute_vscode_unlocked(self, code: str) -> ScriptResult:
        """
        Execute Python code through the stock Isaac Sim Python socket.

        Args:
            code: Python source code to execute.

        Returns:
            ScriptResult with execution output.

        Raises:
            ConnectionRefusedError: Cannot reach Isaac Sim.
            TimeoutError: Connection or read timed out.
        """
        if self._detected_socket_protocol is None:
            self._detected_socket_protocol = await self._detect_socket_protocol()
        half_close = self._detected_socket_protocol == "python_server"
        payload = code
        if self._auth_token:
            payload = f"{PYTHON_SERVER_TOKEN_HEADER} {self._auth_token}\n{code}"
        raw_response = await self._socket_round_trip(
            payload.encode("utf-8"), half_close=half_close
        )
        return self._parse_response(raw_response)

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

    async def _socket_round_trip(self, payload: bytes, *, half_close: bool) -> str:
        """Send one request on the stock socket and read the reply until close.

        Args:
            payload: Bytes to send.
            half_close: Send TCP EOF after the payload. Required by the 6.0
                python_server, which buffers until EOF; fatal on the 5.x VS Code
                extension for scripts that await across frames.

        Returns:
            The raw UTF-8 reply.
        """
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
            if self._socket_protocol == "auto":
                self._detected_socket_protocol = None
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
            deadline: float = time.monotonic() + self._timeout_seconds
            while True:
                remaining: float = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Overall read from Isaac Sim at {self.vscode_address} timed out "
                        f"after {self._timeout_seconds}s."
                    )
                chunk = await asyncio.wait_for(
                    reader.read(self._read_buffer_size),
                    timeout=remaining,
                )
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

    async def _execute_bridge_script(self, code: str) -> ScriptResult:
        """Execute Python code through the custom bridge extension."""
        response = await self._bridge_request("execute_script", {"code": code})
        if response.get("status") == "ok":
            payload = response.get("payload", {})
            return ScriptResult(
                success=True,
                output=str(payload.get("output", "")),
                transport="bridge",
            )

        error = response.get("error", {})
        return ScriptResult(
            success=False,
            output=str(response.get("payload", {}).get("output", "")),
            error_name=str(error.get("name", "BridgeError")),
            error_value=str(error.get("message", "Bridge request failed")),
            traceback=str(error.get("traceback", "")),
            transport="bridge",
        )

    async def ping(self) -> bool:
        """
        Check if Isaac Sim is reachable by executing a trivial script.

        Returns:
            True if Isaac Sim responded successfully, False otherwise.
        """
        async with self._lock:
            if self._prefer_bridge:
                try:
                    response = await self._bridge_request("ping", {})
                    if response.get("status") == "ok":
                        return bool(response.get("payload", {}).get("reachable", True))
                except (ConnectionRefusedError, TimeoutError, OSError, ValueError):
                    if not self._fallback_to_vscode:
                        return False
            try:
                result = await self._execute_vscode_unlocked("print('pong')")
                return result.success and "pong" in result.output
            except (ConnectionRefusedError, TimeoutError, OSError):
                return False

    async def bridge_request(
        self, action: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send a typed request to the bridge transport without fallback."""
        async with self._lock:
            return await self._bridge_request(action, payload or {})

    async def _bridge_request(
        self, action: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send a typed request to the custom bridge extension."""
        if not self._bridge_configured:
            raise ConnectionRefusedError("Bridge transport is not configured.")

        request: Dict[str, Any] = {
            "protocol_version": BRIDGE_PROTOCOL_VERSION,
            "request_id": str(uuid.uuid4()),
            "action": action,
            "payload": payload,
        }
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
                    timeout=self._bridge_timeout_seconds,
                )
                payload_size = struct.unpack(">I", header)[0]
                if payload_size > self._max_response_bytes:
                    raise ValueError(
                        f"Bridge response exceeded {self._max_response_bytes} bytes limit."
                    )
                response_bytes = await asyncio.wait_for(
                    reader.readexactly(payload_size),
                    timeout=self._bridge_timeout_seconds,
                )
            except asyncio.IncompleteReadError as exc:
                raise ConnectionError(
                    f"Bridge at {self.bridge_endpoint} closed connection before sending full response."
                ) from exc

            response: Dict[str, Any] = json.loads(response_bytes.decode("utf-8"))
            if not isinstance(response, dict):
                raise ValueError("Bridge response must be a JSON object.")

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
