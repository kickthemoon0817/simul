"""Local authenticated bridge protocol; imports only the Python standard library."""

from __future__ import annotations

import json
import os
import socket
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

# Bump whenever the add-on and the server stop agreeing on a call shape. The
# add-on is a separately installed ZIP that bundles its own copy of
# BlenderRuntimeSession, so an upgraded server can otherwise send arguments
# (e.g. capture ``agent_id``) that the stale add-on rejects with a bare
# TypeError. Version 2: capture tools take ``agent_id``.
PROTOCOL_VERSION = 2
MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def protocol_mismatch_message(addon: Any, server: Any) -> str:
    """Explain a server/add-on version skew with the fix, not just the symptom."""
    return (
        f"Blender bridge protocol mismatch: the Blender add-on speaks protocol {addon!r}, "
        f"simul-mcp speaks protocol {server!r}. Rebuild the add-on with "
        "`simul blender install-bridge`, reinstall and enable it in Blender, restart "
        "Blender, then run `simul blender attach` again."
    )


class BridgeRemoteError(RuntimeError):
    """A rejected operation, with the remote policy diagnostic preserved."""

    def __init__(self, message: str, remote_type: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.remote_type = remote_type
        self.details = details


class BridgeFiles:
    """Private discovery and attachment files shared by Blender and the CLI."""

    @staticmethod
    def write(path: Path, payload: dict[str, Any]) -> None:
        """Atomically publish credentials readable only by their owner."""
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(prefix=".simul-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(payload, stream)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def read(path: Path) -> dict[str, Any]:
        """Read a regular, owner-only file without following symlinks."""
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor) as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"Not a regular bridge file: {path}")
            if hasattr(os, "getuid") and (
                metadata.st_uid != os.getuid() or metadata.st_mode & 0o077
            ):
                raise PermissionError(
                    f"Bridge file must be owned by this user and mode 0600: {path}"
                )
            data = json.loads(stream.read(MAX_MESSAGE_BYTES + 1))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid bridge file: {path}")
        return data


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
        if endpoint.get("protocol") != PROTOCOL_VERSION:
            raise ValueError(
                protocol_mismatch_message(endpoint.get("protocol"), PROTOCOL_VERSION)
            )
        if endpoint.get("host") != "127.0.0.1":
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
