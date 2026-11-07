"""Local authenticated bridge protocol; imports only the Python standard library."""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Optional

# Re-exported: the Blender add-on and the CLI import BridgeFiles from here.
from ..utils.private_files import BridgeFiles  # noqa: F401

# Bump whenever the add-on and the server stop agreeing on a call's shape or meaning. The
# add-on is a separately installed ZIP that bundles its own copy of
# BlenderRuntimeSession, so an upgraded server can otherwise send arguments
# (e.g. capture ``agent_id``) that the stale add-on rejects with a bare
# TypeError. Version 2: capture tools take ``agent_id``.
# Version 3: capture tools take ``method``; execute_script takes ``max_output_chars``.
# Version 4: focus_on_object's distance_factor is a multiple of the fitting distance; camera tools report world space.
# Version 5: open_blend_file takes discard_unsaved and refuses to discard unsaved changes, which
# agent edits now count as. An older add-on would reject the argument, or, if it were left out,
# open over the unsaved work without the check.
# Version 6: export_file takes object, modifier, animation and format options; import_file takes scale.
PROTOCOL_VERSION = 6
MAX_MESSAGE_BYTES = 32 * 1024 * 1024

#: Capability an add-on lists under ``capabilities`` in its discovery record:
#: given a request with ``"ack": true``, it sends ``{"request_id": ..., "started":
#: true}`` on that connection before it runs the operation, and it runs nothing
#: it has not acknowledged. A connection that ends before the acknowledgement
#: therefore proves the request did not run. Optional, so add-ons and servers of
#: the same protocol version without it still talk to each other.
START_ACK = "start_ack"
#: Seconds a client keeps listening for the acknowledgement after its deadline:
#: the add-on starts nothing after the deadline, and an acknowledgement sent just
#: before it is still on its way.
ACK_GRACE_SECONDS = 0.5
# Pauses between attempts of a request the bridge provably did not run.
_RETRY_FIRST_PAUSE = 0.05
_RETRY_MAX_PAUSE = 0.5


def protocol_mismatch_message(addon: Any, server: Any) -> str:
    """Explain a server/add-on version skew with the fix, not just the symptom."""
    return (
        f"Blender bridge protocol mismatch: the Blender add-on speaks protocol {addon!r}, "
        f"simul speaks protocol {server!r}. Rebuild the add-on with "
        "`simul blender install-bridge`, reinstall and enable it in Blender, restart "
        "Blender, then run `simul blender attach` again."
    )


class BridgeRemoteError(RuntimeError):
    """A rejected operation, with the remote policy diagnostic preserved."""

    def __init__(self, message: str, remote_type: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.remote_type = remote_type
        self.details = details

    @property
    def error_type(self) -> str:
        """The class of the exception raised inside Blender, which error envelopes report."""
        return self.remote_type


class MessageTooLarge(ValueError):
    """A bridge message over MAX_MESSAGE_BYTES, which neither side reads."""

    def __init__(self, size: int, hint: str = "") -> None:
        super().__init__(
            f"Blender bridge message is {size / 2**20:.1f} MiB; the bridge carries at most "
            f"{MAX_MESSAGE_BYTES // 2**20} MiB per call{hint}"
        )
        self.size = size


class BridgeTimeout(TimeoutError):
    """No reply before the deadline.

    Attributes:
        executed: False when the bridge provably did not run the request, True
            when it acknowledged starting it, None when neither is known (an
            add-on without START_ACK).
    """

    def __init__(self, message: str, executed: Optional[bool]) -> None:
        super().__init__(message)
        self.executed = executed


class BridgeConnectionLost(ConnectionError):
    """The connection ended before the reply; ``executed`` as for BridgeTimeout."""

    def __init__(self, message: str, executed: Optional[bool]) -> None:
        super().__init__(message)
        self.executed = executed


class _Lines:
    """Reads newline-terminated messages from a socket against a deadline."""

    def __init__(self, connection: socket.socket) -> None:
        self.connection = connection
        self.buffer = bytearray()

    def next(self, deadline: float) -> Optional[bytes]:
        """Return the next line, or None once ``deadline`` (monotonic) passes."""
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self.connection.settimeout(remaining)
            try:
                chunk = self.connection.recv(65536)
            except (TimeoutError, socket.timeout):
                return None
            if not chunk:
                raise ConnectionError("Blender bridge closed without a response")
            self.buffer.extend(chunk)
            if len(self.buffer) > MAX_MESSAGE_BYTES:
                raise ValueError("Blender bridge response exceeds 32 MiB")
        line, _, rest = bytes(self.buffer).partition(b"\n")
        self.buffer = bytearray(rest)
        return line


class BridgeWire:
    """One newline-delimited JSON request and response per TCP connection."""

    @staticmethod
    def encode(payload: dict[str, Any]) -> bytes:
        """Encode one bounded message.

        Raises:
            MessageTooLarge: Over MAX_MESSAGE_BYTES.
        """
        encoded = json.dumps(payload, allow_nan=False).encode("utf-8") + b"\n"
        if len(encoded) > MAX_MESSAGE_BYTES:
            raise MessageTooLarge(len(encoded))
        return encoded

    @staticmethod
    def request(
        endpoint: dict[str, Any], payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        """Send a request; send it again only while the bridge provably has not run it.

        Against an add-on with START_ACK, a connection that ends before the
        start acknowledgement (the bridge was holding too many connections, or
        restarted) is retried until ``timeout`` runs out. A request that was not
        sent whole is retried against any add-on, since the bridge runs only
        complete requests. Nothing else is retried: an operation whose outcome is
        unknown must not run twice.

        Raises:
            BridgeTimeout: No reply within ``timeout``.
            BridgeConnectionLost: The connection ended before the reply.
            ConnectionRefusedError: Nothing listens on the endpoint.
            BridgeRemoteError: The add-on refused or failed the request.
        """
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
        acknowledged = START_ACK in (endpoint.get("capabilities") or ())
        request = {**payload, "token": endpoint["token"], "protocol": PROTOCOL_VERSION}
        if acknowledged:
            request["ack"] = True
        try:
            message = BridgeWire.encode(request)
        except MessageTooLarge as exc:
            raise MessageTooLarge(
                exc.size,
                ", so the request was not sent. Send less per call, for example a large "
                "mesh as several objects, or write the data to a file and load it in Blender",
            ) from None
        deadline = time.monotonic() + timeout
        pause = _RETRY_FIRST_PAUSE
        while True:
            try:
                response = BridgeWire._exchange(port, message, deadline, acknowledged)
                break
            except BridgeConnectionLost as exc:
                if exc.executed is not False or deadline - time.monotonic() <= pause:
                    raise
            time.sleep(pause)
            pause = min(pause * 2, _RETRY_MAX_PAUSE)
        if not isinstance(response, dict):
            raise ValueError("Invalid Blender bridge response")
        # The bridge refuses a request it cannot parse before it knows its id.
        unread = response.get("request_id") is None and response.get("success") is False
        if response.get("request_id") != payload.get("request_id") and not unread:
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

    @staticmethod
    def _exchange(
        port: int, message: bytes, deadline: float, acknowledged: bool
    ) -> Any:
        """One attempt: connect, send, and read lines until the reply.

        Returns:
            The decoded reply.
        """
        sent = started = False

        def executed() -> Optional[bool]:
            if not sent:
                return False
            return started if acknowledged else None

        try:
            remaining = max(deadline - time.monotonic(), 0.001)
            with socket.create_connection(
                ("127.0.0.1", port), timeout=remaining
            ) as connection:
                connection.settimeout(max(deadline - time.monotonic(), 0.001))
                connection.sendall(message)
                sent = True
                lines = _Lines(connection)
                while True:
                    line = lines.next(deadline)
                    if line is None and acknowledged and not started:
                        # An acknowledgement sent just before the deadline may be
                        # in flight; the bridge starts nothing after it.
                        line = lines.next(time.monotonic() + ACK_GRACE_SECONDS)
                        if line is not None and _is_ack(json.loads(line)):
                            started = True
                            line = None
                    if line is None:
                        raise BridgeTimeout(
                            "Blender bridge response deadline exceeded", executed()
                        )
                    response = json.loads(line)
                    if not _is_ack(response):
                        return response
                    started = True
        except BridgeTimeout:
            raise
        except ConnectionRefusedError:
            raise
        except (TimeoutError, socket.timeout) as exc:
            raise BridgeTimeout(
                "Blender bridge did not accept the request before the deadline", executed()
            ) from exc
        except OSError as exc:
            raise BridgeConnectionLost(
                f"Blender bridge connection ended before the reply ({exc})", executed()
            ) from exc


def _is_ack(message: Any) -> bool:
    """Whether a decoded line is the bridge's start acknowledgement rather than its reply."""
    return (
        isinstance(message, dict)
        and message.get("started") is True
        and "success" not in message
    )
