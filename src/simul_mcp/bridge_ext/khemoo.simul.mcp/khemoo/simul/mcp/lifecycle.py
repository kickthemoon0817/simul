"""Async socket lifecycle for the Simul Isaac bridge."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import struct
import tempfile
from typing import Any, Awaitable, Callable

from .protocol import BridgeRequest, BridgeResponse


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
    ) -> None:
        self._host = host
        self._port = port
        self._request_handler = request_handler
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        self._max_port_retries = max_port_retries
        self._vscode_handler = vscode_handler
        self._request_timeout = request_timeout
        self._server: asyncio.AbstractServer | None = None
        self._actual_port: int = port
        self._discovery_file: str | None = None

    @property
    def address(self) -> str:
        """Return the bound bridge address."""
        return f"{self._host}:{self._actual_port}"

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
                self._actual_port = candidate_port
                return
            except OSError as exc:
                last_error = exc
                continue
        raise OSError(
            f"Failed to bind bridge on ports {self._port}\u2013{self._port + self._max_port_retries - 1}: {last_error}"
        )

    async def stop(self) -> None:
        """Stop the bridge TCP server."""
        if self._server is None:
            return
        self.remove_discovery_file()
        self._server.close()
        await self._server.wait_closed()
        self._server = None

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
            "host": self._host,
            "port": self._actual_port,
            "configured_port": self._port,
        }
        if vscode_port is not None:
            data["vscode_port"] = vscode_port
        fd, tmp = tempfile.mkstemp(dir=discovery_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
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
            request_bytes = await asyncio.wait_for(
                reader.readexactly(payload_size),
                timeout=30.0,
            )
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
