"""Async socket lifecycle for the Simul Isaac bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import struct
import tempfile
import uuid
from typing import Any, Awaitable, Callable

# Stdlib on purpose: this module must stay importable without Kit, and inside
# Kit these records land in the app log through the root handlers.
logger = logging.getLogger(__name__)

from .protocol import BridgeRequest, BridgeResponse


class _RequestReadTimeout(Exception):
    """The client stalled mid-request; not the in-sim handler's fault."""


class BridgeServerLifecycle:
    """Manage the typed TCP bridge server lifecycle."""

    def __init__(
        self,
        host: str,
        port: int,
        request_handler: Callable[[BridgeRequest], Awaitable[BridgeResponse]],
        max_request_bytes: int = 1024 * 1024,
        max_response_bytes: int = 10 * 1024 * 1024,
        max_port_retries: int = 10,
        vscode_handler: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
        request_timeout: float = 120.0,
        socket_path: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._request_handler = request_handler
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        self._max_port_retries = max_port_retries
        self._vscode_handler = vscode_handler
        self._request_timeout = request_timeout
        self._socket_path = socket_path
        self._actual_socket_path: str | None = None
        self._socket_inode: tuple[int, int] | None = None
        self._server: asyncio.AbstractServer | None = None
        self._unix_server: asyncio.AbstractServer | None = None
        self._actual_port: int = port
        self._discovery_file: str | None = None

    @property
    def address(self) -> str:
        """Return the bound bridge address."""
        return f"{self._host}:{self._actual_port}"

    @property
    def actual_socket_path(self) -> str | None:
        """Return the Unix socket path actually bound, when one is serving."""
        return self._actual_socket_path

    @property
    def actual_port(self) -> int:
        """Return the port the server actually bound to."""
        return self._actual_port

    async def start(self) -> None:
        """Start the bridge TCP server, retrying on successive ports if the configured one is busy."""
        if self._server is not None:
            return
        last_error: OSError | None = None
        for attempt in range(self._max_port_retries):
            candidate_port = self._port + attempt
            if candidate_port > 65535:
                break
            try:
                self._server = await asyncio.start_server(
                    self._handle_client,
                    host=self._host,
                    port=candidate_port,
                    family=socket.AF_INET,
                )
                await self._server.start_serving()
                # For port 0 the OS picks; report what was actually bound.
                self._actual_port = self._server.sockets[0].getsockname()[1]
                await self._start_unix()
                return
            except OSError as exc:
                last_error = exc
                continue
        raise OSError(
            f"Failed to bind bridge on ports {self._port}\u2013{self._port + self._max_port_retries - 1}: {last_error}"
        )

    async def _start_unix(self) -> None:
        """Serve the same protocol on a Unix socket, when one is configured.

        The socket sits on the discovery volume, which container and host
        already share, so the host reaches the bridge without a published
        port. Failure here is logged and non-fatal: the TCP listener is
        already up, and a container without the volume mounted should not
        lose its working transport over the optional one.
        """
        if not self._socket_path or self._unix_server is not None:
            return
        try:
            path = self._socket_path
            if os.path.exists(path):
                if await self._socket_is_live(path):
                    # Another instance is serving here — typically a second
                    # container sharing the discovery volume, where every
                    # bridge is pid 1 and ships the same configured name.
                    # Stealing the name would hijack its traffic and, on
                    # stop, destroy its socket. Bind a unique sibling and
                    # advertise that instead.
                    stem, ext = os.path.splitext(path)
                    path = f"{stem}-{uuid.uuid4().hex[:8]}{ext or '.sock'}"
                else:
                    # A crash left the inode behind; bind() would refuse it.
                    os.unlink(path)
            self._unix_server = await asyncio.start_unix_server(
                self._handle_client, path=path
            )
            await self._unix_server.start_serving()
            # connect(2) needs write permission on the socket inode. In a
            # container this process is root, so the default mode would leave
            # the host user unable to connect — the 0600 discovery-file
            # failure again, one layer down.
            os.chmod(path, 0o666)
            self._actual_socket_path = path
            info = os.stat(path)
            self._socket_inode = (info.st_dev, info.st_ino)
        except OSError as exc:
            logger.warning("Bridge Unix socket %s unavailable: %s", self._socket_path, exc)
            self._unix_server = None
            self._actual_socket_path = None
            self._socket_inode = None

    @staticmethod
    async def _socket_is_live(path: str) -> bool:
        """Return True when something is accepting connections at ``path``."""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(path), timeout=0.5
            )
        except (OSError, asyncio.TimeoutError):
            return False
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
        return True

    async def stop(self) -> None:
        """Stop the bridge TCP server."""
        if self._server is None:
            return
        self.remove_discovery_file()
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        if self._unix_server is not None:
            self._unix_server.close()
            await self._unix_server.wait_closed()
            self._unix_server = None
        if self._actual_socket_path and self._socket_inode is not None:
            # Only remove the inode this instance bound. Another bridge may
            # have replaced the name since; deleting its live socket would cut
            # it off with nothing in any log.
            try:
                info = os.stat(self._actual_socket_path)
                if (info.st_dev, info.st_ino) == self._socket_inode:
                    os.unlink(self._actual_socket_path)
            except OSError:
                pass
        self._actual_socket_path = None
        self._socket_inode = None

    # A wildcard bind covers every interface, loopback included, but it is not
    # an address anything can connect to — and the reader drops a discovery
    # entry whose host is not loopback. Advertise the address the wildcard
    # already answers on, so a container that binds 0.0.0.0 (which is what
    # Docker's published port requires) still shows up.
    _WILDCARD_BINDS = frozenset({"", "0.0.0.0", "::", "*"})

    def _advertised_host(self) -> str:
        """Return an address a client can actually dial."""
        if self._host in self._WILDCARD_BINDS:
            return "127.0.0.1"
        return self._host

    def write_discovery_file(
        self,
        discovery_dir: str,
        pid: int,
        vscode_port: int | None = None,
    ) -> None:
        """Write a discovery file with the actual bound port."""
        os.makedirs(discovery_dir, mode=0o700, exist_ok=True)
        filepath = os.path.join(discovery_dir, f"simul-mcp-{pid}.json")
        data = {
            "pid": pid,
            "host": self._advertised_host(),
            "port": self._actual_port,
            "configured_port": self._port,
        }
        if vscode_port is not None:
            data["vscode_port"] = vscode_port
        if self._actual_socket_path and self._unix_server is not None:
            data["socket_path"] = self._actual_socket_path
        fd, tmp = tempfile.mkstemp(dir=discovery_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            # mkstemp creates 0600. In a container the bridge runs as root
            # (the Isaac Sim image keeps /isaac-sim mode 0750 root:root), so a
            # 0600 file on the shared discovery volume is unreadable by the MCP
            # server running as an ordinary user on the host — discovery finds
            # nothing even though the bridge is up. The contents are a pid, a
            # host and two ports; the directory stays 0700, so this only widens
            # access where the directory was deliberately shared.
            os.chmod(tmp, 0o644)
            os.rename(tmp, filepath)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        self._discovery_file = filepath

    def remove_discovery_file(self) -> None:
        """Remove the discovery file if it exists."""
        if self._discovery_file:
            try:
                os.remove(self._discovery_file)
            except OSError:
                pass
            self._discovery_file = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Detect protocol and route to bridge or VS Code compat handler."""
        try:
            peek = await asyncio.wait_for(reader.readexactly(4), timeout=5.0)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError):
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return

        # Try to interpret as a bridge length prefix
        candidate_size = struct.unpack(">I", peek)[0]
        if 0 < candidate_size <= self._max_request_bytes:
            # Likely bridge protocol -- read the payload
            await self._handle_bridge_client(reader, writer, peek, candidate_size)
        else:
            # Likely raw Python code -- VS Code compat
            await self._handle_vscode_compat_client(reader, writer, peek)

    async def _handle_bridge_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        header: bytes,
        payload_size: int,
    ) -> None:
        """Handle a bridge-protocol client (length-prefixed JSON)."""
        request_id = ""
        try:
            try:
                request_bytes = await asyncio.wait_for(
                    reader.readexactly(payload_size),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                # Distinguished from the handler timeout below: this one means
                # the client sent a length prefix and then stalled. Reporting it
                # as "the handler is still running inside Isaac Sim" would point
                # the operator at the wrong subsystem entirely.
                raise _RequestReadTimeout(
                    f"Client sent {payload_size} bytes of header then stalled; "
                    "no request body within 30.0s."
                ) from None
            request = BridgeRequest.from_json(request_bytes)
            request_id = request.request_id
            # Without this, a script that never returns hangs Kit permanently:
            # the handler runs on the main thread, so nothing else in the app
            # makes progress and SIGKILL is the only way out. Cutting the wait
            # does not stop the runaway work, but it releases the client and
            # makes the failure visible instead of silent.
            response = await asyncio.wait_for(
                self._request_handler(request), timeout=self._request_timeout
            )
        except _RequestReadTimeout as exc:
            response = BridgeResponse.failure(
                request_id=request_id,
                name="RequestReadTimeout",
                message=str(exc),
            )
        except asyncio.TimeoutError:
            response = BridgeResponse.failure(
                request_id=request_id,
                name="RequestTimeout",
                message=(
                    f"Handler exceeded {self._request_timeout}s. The action may "
                    "still be running inside Isaac Sim."
                ),
            )
        except Exception as exc:
            response = BridgeResponse.failure(
                request_id=request_id,
                name=type(exc).__name__,
                message=str(exc),
            )
        try:
            await self._write_frame(writer, self._serialise_response(response))
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_vscode_compat_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        initial_bytes: bytes,
    ) -> None:
        """Handle a VS Code-protocol client (raw Python code, JSON response)."""
        try:
            # Read remaining code bytes
            chunks: list[bytes] = [initial_bytes]
            total = len(initial_bytes)
            while True:
                try:
                    chunk = await asyncio.wait_for(reader.read(1024 * 1024), timeout=5.0)
                except asyncio.TimeoutError:
                    break
                if not chunk:
                    break
                total += len(chunk)
                if total > self._max_request_bytes:
                    break
                chunks.append(chunk)

            code = b"".join(chunks).decode("utf-8", errors="replace")

            if self._vscode_handler is not None:
                result = await self._vscode_handler(code)
            else:
                result = {"status": "error", "output": "VS Code compat handler not configured."}

            response_bytes = json.dumps(result, separators=(",", ":")).encode("utf-8")
            writer.write(response_bytes)
            await writer.drain()
        except Exception:
            try:
                error = json.dumps({"status": "error", "output": "Internal error"}).encode("utf-8")
                writer.write(error)
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_frame(self, reader: asyncio.StreamReader) -> bytes:
        """Read a length-prefixed request frame."""
        header = await reader.readexactly(4)
        size = struct.unpack(">I", header)[0]
        if size > self._max_request_bytes:
            raise ValueError(
                f"Bridge request exceeded {self._max_request_bytes} bytes."
            )
        return await reader.readexactly(size)

    async def _write_frame(
        self, writer: asyncio.StreamWriter, payload: bytes
    ) -> None:
        """Write a length-prefixed response frame."""
        if len(payload) > self._max_response_bytes:
            raise ValueError(
                f"Bridge response exceeded {self._max_response_bytes} bytes."
            )
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()

    def _serialise_response(self, response: BridgeResponse) -> bytes:
        """Serialize a response and downgrade oversize payloads to a fixed error."""
        payload = response.to_json()
        if len(payload) <= self._max_response_bytes:
            return payload

        fallback = BridgeResponse.failure(
            request_id=response.request_id,
            name="ResponseTooLarge",
            message=(
                "Bridge response exceeded "
                f"{self._max_response_bytes} bytes and was truncated."
            ),
        ).to_json()
        if len(fallback) > self._max_response_bytes:
            raise ValueError(
                "Bridge fallback error response exceeded the configured size limit."
            )
        return fallback
