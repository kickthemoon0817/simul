"""Local authenticated bridge protocol; imports only the Python standard library."""

from __future__ import annotations

import json
import socket
import time
from typing import Any

# Re-exported: the Blender add-on and the CLI import BridgeFiles from here.
from ..utils.private_files import BridgeFiles  # noqa: F401

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 32 * 1024 * 1024


class BridgeRemoteError(RuntimeError):
    """A rejected operation, with the remote policy diagnostic preserved."""

    def __init__(self, message: str, remote_type: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.remote_type = remote_type
        self.details = details


class BridgeWire:
    """One newline-delimited JSON request and response per TCP connection."""

    @staticmethod
    def encode(payload: dict[str, Any]) -> bytes:
        """Encode one bounded message."""
        encoded = json.dumps(payload, allow_nan=False).encode("utf-8") + b"\n"
        if len(encoded) > MAX_MESSAGE_BYTES:
            raise ValueError("Blender bridge message exceeds 32 MiB")
        return encoded

    @staticmethod
    def request(
        endpoint: dict[str, Any], payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        """Send exactly once; never retry an operation whose outcome is unknown."""
        if (
            endpoint.get("protocol") != PROTOCOL_VERSION
            or endpoint.get("host") != "127.0.0.1"
        ):
            raise ValueError("Unsupported Blender bridge endpoint")
        if not isinstance(endpoint.get("token"), str) or not endpoint["token"]:
            raise ValueError("Missing Blender bridge authentication token")
        port = endpoint.get("port")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid Blender bridge port")
        request = {**payload, "token": endpoint["token"], "protocol": PROTOCOL_VERSION}
        deadline = time.monotonic() + timeout
        with socket.create_connection(
            ("127.0.0.1", port), timeout=timeout
        ) as connection:
            connection.settimeout(timeout)
            connection.sendall(BridgeWire.encode(request))
            buffer = bytearray()
            while b"\n" not in buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Blender bridge response deadline exceeded")
                connection.settimeout(remaining)
                chunk = connection.recv(65536)
                if not chunk:
                    raise ConnectionError("Blender bridge closed without a response")
                buffer.extend(chunk)
                if len(buffer) > MAX_MESSAGE_BYTES:
                    raise ValueError("Blender bridge response exceeds 32 MiB")
        response = json.loads(buffer.split(b"\n", 1)[0])
        if not isinstance(response, dict):
            raise ValueError("Invalid Blender bridge response")
        if response.get("request_id") != payload.get("request_id"):
            raise ValueError("Blender bridge response identity mismatch")
        if response.get("success") is not True:
            raise BridgeRemoteError(
                response.get("error", "Blender bridge request failed"),
                response.get("error_type", "RuntimeError"),
                response.get("details", {}),
            )
        result = response["result"]
        if not isinstance(result, dict):
            raise ValueError("Invalid Blender bridge result")
        return result
