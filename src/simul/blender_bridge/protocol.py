"""Local authenticated bridge protocol; imports only the Python standard library."""

from __future__ import annotations

import json
import socket
import time
from typing import Any

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
# Version 7: bake_simulation and free_bake take scope, and a bake replaces an earlier one;
# get_physics_state takes step_simulation; get/set_rigid_body_world are new. An older add-on
# would reject the arguments, or, left without them, report a re-bake it did not do.
# Version 8: export_simready_usd takes include_children, include_lights_and_cameras and force, and
# authors UsdPhysics and UsdSemantics schemas through the add-on's bundled simready_usd module.
# Version 9: capture tools take camera_name; capture_viewport_sequence writes files (output_dir, max_frames).
# Version 10: setup_rigid_body, add_force_field and add_rigid_body_constraint take new settings
# (kinematic, direction, axis, limits, springs, motors, ...) and check them up front;
# bake_simulation takes allow_long_bake and refuses a bake over its work budget without it.
PROTOCOL_VERSION = 10
MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def protocol_mismatch_message(addon: Any, server: Any) -> str:
    """Explain a server/add-on version skew with the fix, not just the symptom."""
    return (
        f"Blender bridge protocol mismatch: the Blender add-on speaks protocol {addon!r}, "
        f"simul speaks protocol {server!r}. Rebuild the add-on with "
        "`simul blender install-bridge`, reinstall and enable it in Blender, restart "
        "Blender, then run `simul blender attach` again."
    )


class BridgeProtocolError(ValueError):
    """The bridge endpoint or its reply does not follow this protocol.

    Another protocol version, an endpoint file that is not this machine's
    bridge, a reply that is malformed or answers another request. Not the
    caller's input, though a ValueError as before: ``error_type`` names it.
    """

    error_type = "BridgeProtocolError"


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
            raise BridgeProtocolError(
                protocol_mismatch_message(endpoint.get("protocol"), PROTOCOL_VERSION)
            )
        if endpoint.get("host") != "127.0.0.1":
            raise BridgeProtocolError("Unsupported Blender bridge endpoint")
        if not isinstance(endpoint.get("token"), str) or not endpoint["token"]:
            raise BridgeProtocolError("Missing Blender bridge authentication token")
        port = endpoint.get("port")
        if type(port) is not int or not 1 <= port <= 65535:
            raise BridgeProtocolError("Invalid Blender bridge port")
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
                    raise BridgeProtocolError("Blender bridge response exceeds 32 MiB")
        try:
            response = json.loads(buffer.split(b"\n", 1)[0])
        except ValueError as exc:
            raise BridgeProtocolError(f"Undecodable Blender bridge response: {exc}") from exc
        if not isinstance(response, dict):
            raise BridgeProtocolError("Invalid Blender bridge response")
        if response.get("request_id") != payload.get("request_id"):
            raise BridgeProtocolError("Blender bridge response identity mismatch")
        if response.get("success") is not True:
            raise BridgeRemoteError(
                response.get("error", "Blender bridge request failed"),
                response.get("error_type", "RuntimeError"),
                response.get("details", {}),
            )
        result = response["result"]
        if not isinstance(result, dict):
            raise BridgeProtocolError("Invalid Blender bridge result")
        return result
