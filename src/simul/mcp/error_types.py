"""The ``error_type`` values a tool's error envelope carries.

Every failed tool call returns ``{"success": false, "error_type": ...,
"error": ..., "details": ...}`` and is sent with ``isError: true``. An agent
branches on ``error_type``, so the classes below mean the same thing on every
backend and in every layer (schema validation, request models, adapters,
transports). Backends keep more specific domain types where they say more
than a class would (``PIEActive``, ``RefusedOperation``, ``CaptureError``);
transport failures never leak a library's exception name
(``ClientConnectorError``, ``ConnectionResetError``, ``EmptyResponse``): they
are mapped onto ``ConnectionError`` or ``TimeoutError``, with the original
name kept in ``details.cause``.

The taxonomy is documented in ``docs/tools.md`` (Errors).
"""

from __future__ import annotations

import errno
from typing import Any, Dict, Optional

# -- Classes every backend shares ----------------------------------------------

#: The arguments do not fit the tool: a wrong type, a missing or unexpected
#: argument, a value out of range or not among the choices. Raised by the MCP
#: schema, a request model or the tool itself; ``details.errors`` names each
#: offending field. Fix the call; retrying it unchanged fails again.
VALIDATION_ERROR = "ValidationError"

#: The thing the call names (a prim, an actor, a stage, an instance, a tool)
#: does not exist.
NOT_FOUND_ERROR = "NotFoundError"

#: The engine could not be reached, or the connection broke before it answered
#: (refused, reset, closed without a reply). The operation may not have run.
CONNECTION_ERROR = "ConnectionError"

#: The engine did not answer in time. The operation may still be running.
TIMEOUT_ERROR = "TimeoutError"

#: The engine received the request and reported a failure the tool does not
#: name more precisely; ``error`` carries the engine's message.
ENGINE_ERROR = "EngineError"

#: The instance is serving another call; retry shortly. It is running, not
#: unreachable.
INSTANCE_BUSY = "InstanceBusy"

#: A file path or URL is outside ``security.sandbox`` (``details`` names the
#: allowed roots and schemes).
SANDBOX_ERROR = "SandboxError"

# -- Domain types --------------------------------------------------------------

#: The calling agent spent its rate-limit budget; ``retry_after_seconds`` says
#: when a token is available.
RATE_LIMIT_ERROR = "RateLimitError"

#: Another agent holds a live claim on the Isaac instance and
#: ``isaac_sim.enforce_claims`` is on; ``details`` names the holder.
INSTANCE_CLAIMED = "InstanceClaimed"

#: The instance claim records could not be read or saved.
CLAIM_STORE_ERROR = "ClaimStoreError"

#: The tool keeps state per client and the request carries no client identity
#: (an unauthenticated sessionless HTTP request).
SESSION_REQUIRED = "SessionRequired"

#: The caller may not act on what it named (another agent's claim).
PERMISSION_ERROR = "PermissionError"

#: The tool refuses an operation that would damage the running application or
#: unsaved work unless an override flag is passed.
REFUSED_OPERATION = "RefusedOperation"

#: Unreal is in Play-In-Editor, where the operation is not allowed.
PIE_ACTIVE = "PIEActive"

#: The engine refused the credentials (the python_server token, the Remote
#: Control passphrase).
AUTHENTICATION_ERROR = "AuthenticationError"

#: The backend or its current mode cannot do this operation.
UNSUPPORTED_OPERATION = "UnsupportedOperation"

#: The operation is not possible in the current state (nothing claimed to
#: release, no stage loaded).
INVALID_STATE = "InvalidState"

#: The failure classes, for documentation and tests.
CLASSES: tuple[str, ...] = (
    VALIDATION_ERROR,
    NOT_FOUND_ERROR,
    CONNECTION_ERROR,
    TIMEOUT_ERROR,
    ENGINE_ERROR,
    INSTANCE_BUSY,
    SANDBOX_ERROR,
)

# Library and transport names that mean "could not reach the engine" or "it
# did not answer in time", by the error_type an adapter or exception produced.
_TRANSPORT_ALIASES: Dict[str, str] = {
    # builtins
    "ConnectionRefusedError": CONNECTION_ERROR,
    "ConnectionResetError": CONNECTION_ERROR,
    "ConnectionAbortedError": CONNECTION_ERROR,
    "BrokenPipeError": CONNECTION_ERROR,
    "IncompleteReadError": CONNECTION_ERROR,
    # aiohttp (Unreal Remote Control)
    "ClientConnectorError": CONNECTION_ERROR,
    "ClientConnectorDNSError": CONNECTION_ERROR,
    "ClientConnectionError": CONNECTION_ERROR,
    "ClientConnectionResetError": CONNECTION_ERROR,
    "ClientOSError": CONNECTION_ERROR,
    "ServerConnectionError": CONNECTION_ERROR,
    "ServerDisconnectedError": CONNECTION_ERROR,
    "ServerTimeoutError": TIMEOUT_ERROR,
    "ConnectionTimeoutError": TIMEOUT_ERROR,
    "SocketTimeoutError": TIMEOUT_ERROR,
    # Isaac Sim transports
    "EmptyResponse": CONNECTION_ERROR,
    "BridgeCircuitOpenError": CONNECTION_ERROR,
    "BridgeResponseConnectionError": CONNECTION_ERROR,
    "BridgeResponseTimeoutError": TIMEOUT_ERROR,
}

# The names a payload's error_type may carry and the class they stand for: the
# transports', plus the ValueError tools report for an argument they refuse.
# (Not used on exceptions by class: JSONDecodeError is a ValueError too.)
_PAYLOAD_ALIASES: Dict[str, str] = {**_TRANSPORT_ALIASES, "ValueError": VALIDATION_ERROR}

_CONNECTION_ERRNOS = frozenset(
    code
    for code in (
        getattr(errno, name, None)
        for name in (
            "ECONNREFUSED",
            "ECONNRESET",
            "ECONNABORTED",
            "EPIPE",
            "ENOTCONN",
            "EHOSTUNREACH",
            "ENETUNREACH",
            "EHOSTDOWN",
            "ENETDOWN",
        )
    )
    if code is not None
)

_EMPTY_MESSAGES = {
    TIMEOUT_ERROR: "The engine did not answer in time; the operation may still be running.",
    CONNECTION_ERROR: "The connection to the engine failed before it answered.",
}
_DEFAULT_MESSAGE = "The operation failed without a message."


def canonical_error_type(name: Optional[str]) -> Optional[str]:
    """Return the class a payload's ``error_type`` belongs to, or None to keep ``name``."""
    if not name:
        return None
    return _PAYLOAD_ALIASES.get(name)


def classify_exception(exc: BaseException) -> Optional[str]:
    """Return the class an exception that escaped a tool call belongs to.

    Args:
        exc: The exception.

    Returns:
        ``TimeoutError`` for a timeout, ``ConnectionError`` for a connection
        that failed (by class, by library name, or by a socket ``errno``),
        ``ValidationError`` for a bare ``ValueError`` (an argument check), or
        None when the exception keeps its own name.
    """
    if isinstance(exc, TimeoutError):
        return TIMEOUT_ERROR
    if isinstance(exc, ConnectionError):
        return CONNECTION_ERROR
    for klass in type(exc).__mro__:
        alias = _TRANSPORT_ALIASES.get(klass.__name__)
        if alias is not None:
            return alias
    if isinstance(exc, OSError) and exc.errno in _CONNECTION_ERRNOS:
        return CONNECTION_ERROR
    if type(exc) is ValueError:
        return VALIDATION_ERROR
    return None


def normalize_error_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Give a failed payload a stable ``error_type`` and a message (in place).

    A transport name becomes its class, with the original kept as
    ``details.cause``. A failure without any ``error_type`` gets one:
    ``ConnectionError`` when the payload says the engine is not reachable
    (``reachable`` or ``connected`` false), ``EngineError`` otherwise. A
    ``ValidationError`` without ``details.errors`` gets its message as the one
    error. A failure with an empty message gets one: the ``*_error`` keys of a
    partial failure, or a sentence saying what happened.

    Args:
        payload: A tool payload whose ``success`` is False.

    Returns:
        The same dict.
    """
    raw = payload.get("error_type")
    current = raw if isinstance(raw, str) and raw else None
    canonical = canonical_error_type(current)
    if canonical is not None and canonical != current:
        details = payload.get("details")
        details = dict(details) if isinstance(details, dict) else {}
        details.setdefault("cause", current)
        payload["details"] = details
        payload["error_type"] = canonical
    elif current is None:
        unreachable = payload.get("reachable") is False or payload.get("connected") is False
        payload["error_type"] = CONNECTION_ERROR if unreachable else ENGINE_ERROR
    if payload["error_type"] == VALIDATION_ERROR:
        # Every ValidationError lists its failures, whichever layer raised it.
        details = payload.get("details")
        details = dict(details) if isinstance(details, dict) else {}
        if not isinstance(details.get("errors"), list):
            details["errors"] = [
                {"field": None, "message": str(payload.get("error") or ""), "type": "value_error"}
            ]
            payload["details"] = details
    if not payload.get("error"):
        # A partial failure names what failed under suffixed keys; a tool that
        # set success false itself may have said why under reason or message.
        parts = [
            f"{key}: {value}"
            for key, value in payload.items()
            if key.endswith("_error") and key != "error" and value is not None
        ]
        stated = next(
            (
                payload[key]
                for key in ("reason", "message")
                if isinstance(payload.get(key), str) and payload[key]
            ),
            None,
        )
        payload["error"] = (
            "; ".join(parts)
            or stated
            or _EMPTY_MESSAGES.get(payload["error_type"], _DEFAULT_MESSAGE)
        )
    return payload
