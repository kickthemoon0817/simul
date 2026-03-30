"""Typed request/response protocol for the Simul Isaac bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BridgeError:
    """Structured bridge error payload."""

    name: str
    message: str
    traceback: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error for JSON transport."""
        data = {"name": self.name, "message": self.message}
        if self.traceback:
            data["traceback"] = self.traceback
        return data


@dataclass(frozen=True)
class BridgeRequest:
    """Typed request envelope for the Isaac bridge."""

    request_id: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = "1.0"

    @classmethod
    def from_json(cls, raw: bytes) -> "BridgeRequest":
        """Parse a request envelope from JSON bytes."""
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Bridge request must be a JSON object.")
        payload = data.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError("Bridge request payload must be an object.")
        return cls(
            request_id=str(data.get("request_id", "")),
            action=str(data.get("action", "")),
            payload=payload,
            protocol_version=str(data.get("protocol_version", "1.0")),
        )


@dataclass(frozen=True)
class BridgeResponse:
    """Typed response envelope for the Isaac bridge."""

    request_id: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    error: BridgeError | None = None
    protocol_version: str = "1.0"

    @classmethod
    def success(
        cls, request_id: str, payload: dict[str, Any]
    ) -> "BridgeResponse":
        """Build a successful response envelope."""
        return cls(request_id=request_id, status="ok", payload=payload)

    @classmethod
    def failure(
        cls,
        request_id: str,
        name: str,
        message: str,
        traceback: str = "",
    ) -> "BridgeResponse":
        """Build a failed response envelope."""
        return cls(
            request_id=request_id,
            status="error",
            error=BridgeError(name=name, message=message, traceback=traceback),
        )

    def to_json(self) -> bytes:
        """Serialize the response envelope to JSON bytes."""
        data: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "status": self.status,
            "payload": self.payload,
        }
        if self.error is not None:
            data["error"] = self.error.to_dict()
        return json.dumps(data, separators=(",", ":")).encode("utf-8")
