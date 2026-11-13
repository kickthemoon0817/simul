"""The ``error_type`` values a tool's error envelope carries.

Every failed tool call returns ``{"success": false, "error_type": ...,
"error": ..., "details": ...}`` and is sent with ``isError: true``. An agent
branches on ``error_type``, so the vocabulary below means the same thing on
every backend and in every layer (schema validation, request models,
adapters, transports): seven classes every backend shares, and domain types
where a backend says more than a class would (``PIEActive``,
``RefusedOperation``, ``CaptureError``, the ``*NotApplied`` family).

Nothing outside the vocabulary reaches a client. A library's or Python's
exception name (``ClientConnectorError``, ``ConnectionResetError``,
``KeyError``, an ``AttributeError`` raised inside the engine) is mapped onto
the class of its family by ``family_of``, and the original name is kept in
``details.exception_type``.

The vocabulary is documented in ``docs/tools.md`` (Errors).
"""

from __future__ import annotations

import difflib
import errno
import re
from typing import Any, Dict, List, Optional, Tuple

from ..utils.errors import exception_type_name

# -- Classes every backend shares ----------------------------------------------

#: The arguments do not fit the tool: a wrong type, a missing or unexpected
#: argument, a value out of range or not among the choices. Raised by the MCP
#: schema, a request model or the tool itself; ``details.errors`` names each
#: offending field, and ``details.code`` names the check when a tool has
#: several (USD's ``InvalidPrimPath``). Fix the call; retrying it unchanged
#: fails again.
VALIDATION_ERROR = "ValidationError"

#: The thing the call names (a prim, an actor, an object, a stage, an
#: instance, a tool, a file) does not exist.
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

# -- Domain types: the server ------------------------------------------------

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

#: The caller may not act on what it named: another agent's claim, or a file
#: this process may not read or write (Python's ``PermissionError``).
PERMISSION_DENIED = "PermissionDenied"

#: ``security.allow_script_execution`` is off, so the arbitrary-code tools
#: are not available.
SCRIPT_EXECUTION_DISABLED = "ScriptExecutionDisabled"

#: A script, request or reply is larger than the transport accepts.
PAYLOAD_TOO_LARGE = "PayloadTooLarge"

# -- Domain types: shared by the engines ---------------------------------------

#: The tool refuses an operation that would damage the running application or
#: unsaved work unless an override flag is passed.
REFUSED_OPERATION = "RefusedOperation"

#: The engine refused the credentials (the python_server token, the Remote
#: Control passphrase).
AUTHENTICATION_ERROR = "AuthenticationError"

#: The backend or its current mode cannot do this operation.
UNSUPPORTED_OPERATION = "UnsupportedOperation"

#: The operation is not possible in the current state (nothing claimed to
#: release, no stage loaded, a folder that already holds the import).
INVALID_STATE = "InvalidState"

#: An agent's script raised inside the engine; ``details`` carries the
#: traceback.
SCRIPT_ERROR = "ScriptError"

#: The engine's script printed nothing where the tool expected its JSON reply.
EMPTY_OUTPUT = "EmptyOutput"

#: The asset at the path is not of the kind the tool needs (a level actor
#: where a class is needed, a texture where a material is).
INVALID_ASSET_TYPE = "InvalidAssetType"

#: The file format is not one the tool reads or writes.
UNSUPPORTED_FORMAT = "UnsupportedFormat"

#: A viewport capture failed.
CAPTURE_ERROR = "CaptureError"

#: There is no viewport to read or drive (a headless editor).
VIEWPORT_UNAVAILABLE = "ViewportUnavailable"

#: The editor UI the tool drives is not available (a headless editor).
UI_UNAVAILABLE = "UiUnavailable"

#: The simulation must be running (played or stepped once) for the answer.
SIMULATION_NOT_RUNNING = "SimulationNotRunning"

#: An asset check ran and found errors (``details`` lists the issues).
VALIDATION_FAILED = "ValidationFailed"

#: A SimReady export was stopped by the validation it runs first.
SIMREADY_VALIDATION_FAILED = "SimReadyValidationFailed"

# -- Domain types: one backend -------------------------------------------------

#: Unreal is in Play-In-Editor, where the operation is not allowed.
PIE_ACTIVE = "PIEActive"

#: Unreal could not spawn the actor from the class or asset given.
SPAWN_ERROR = "SpawnError"

#: The optional Unreal plugin the tool needs (USDImporter, ...) is not enabled.
PLUGIN_UNAVAILABLE = "PluginUnavailable"

#: The Unreal body cannot simulate as asked; ``details.problems`` lists why.
INCOMPATIBLE_BODY = "IncompatibleBody"

#: The Unreal actor has no copy in the running simulation (spawned after it
#: started, destroyed in it, or in a World Partition cell it has not streamed).
NOT_IN_SIMULATION = "NotInSimulation"

#: The Unreal editor exited while ``simul unreal setup`` waited for it.
EDITOR_EXITED = "EditorExited"

#: The Isaac Sim bridge extension the tool needs is not configured.
BRIDGE_UNAVAILABLE = "BridgeUnavailable"

#: A bridge (Isaac Sim's or Blender's) answered outside its protocol: another
#: protocol version, a malformed or unmatched reply. The request may have run.
BRIDGE_PROTOCOL_ERROR = "BridgeProtocolError"

#: Blender's bridge could not encode the result; the operation may have run.
RESPONSE_ENCODING_ERROR = "ResponseEncodingError"

#: The attached Blender window could not be used.
ATTACHMENT_ERROR = "AttachmentError"

#: The attached Blender process exited or its bridge stopped listening.
ATTACHMENT_STALE = "AttachmentStale"

#: The attached Blender process loaded another file or lost its window.
ATTACHMENT_TARGET_CHANGED = "AttachmentTargetChanged"

#: Blender refuses to replace a file with unsaved changes without the flag.
UNSAVED_CHANGES = "UnsavedChangesError"

#: Blender refused a key part way through a batch; the keys before it stay
#: (``details`` gives how many went in and the one refused).
KEYFRAMES_PARTLY_INSERTED = "KeyframesPartlyInserted"

#: A headless USD file could not be opened; ``details.reason`` says why.
LOAD_ERROR = "LoadError"

#: A headless USD stage is serving a call that cannot run beside this one;
#: retry shortly.
STAGE_BUSY = "StageBusy"

#: A headless USD computation (bounds, statistics) failed.
COMPUTATION_ERROR = "ComputationError"

#: ``simul logs`` found no log file at the path.
LOG_FILE_NOT_FOUND = "LogFileNotFound"

#: The suffix of the read-back family: the engine took the request, but what
#: it reads back afterwards is not what was asked (``TransformNotApplied``,
#: ``MaterialNotApplied``, ``RenderSettingNotApplied``). The payload keeps the
#: values read back, with ``details.expected`` and ``details.actual`` where
#: the tool reports them.
NOT_APPLIED_SUFFIX = "NotApplied"

#: The classes every backend shares.
CLASSES: Tuple[str, ...] = (
    VALIDATION_ERROR,
    NOT_FOUND_ERROR,
    CONNECTION_ERROR,
    TIMEOUT_ERROR,
    ENGINE_ERROR,
    INSTANCE_BUSY,
    SANDBOX_ERROR,
)

#: The domain types; with ``CLASSES`` and the ``*NotApplied`` family, every
#: ``error_type`` a client can see.
DOMAIN_TYPES: Tuple[str, ...] = (
    RATE_LIMIT_ERROR,
    INSTANCE_CLAIMED,
    CLAIM_STORE_ERROR,
    SESSION_REQUIRED,
    PERMISSION_DENIED,
    SCRIPT_EXECUTION_DISABLED,
    PAYLOAD_TOO_LARGE,
    REFUSED_OPERATION,
    AUTHENTICATION_ERROR,
    UNSUPPORTED_OPERATION,
    INVALID_STATE,
    SCRIPT_ERROR,
    EMPTY_OUTPUT,
    INVALID_ASSET_TYPE,
    UNSUPPORTED_FORMAT,
    CAPTURE_ERROR,
    VIEWPORT_UNAVAILABLE,
    UI_UNAVAILABLE,
    SIMULATION_NOT_RUNNING,
    VALIDATION_FAILED,
    SIMREADY_VALIDATION_FAILED,
    PIE_ACTIVE,
    SPAWN_ERROR,
    PLUGIN_UNAVAILABLE,
    INCOMPATIBLE_BODY,
    NOT_IN_SIMULATION,
    EDITOR_EXITED,
    BRIDGE_UNAVAILABLE,
    BRIDGE_PROTOCOL_ERROR,
    RESPONSE_ENCODING_ERROR,
    ATTACHMENT_ERROR,
    ATTACHMENT_STALE,
    ATTACHMENT_TARGET_CHANGED,
    UNSAVED_CHANGES,
    KEYFRAMES_PARTLY_INSERTED,
    LOAD_ERROR,
    STAGE_BUSY,
    COMPUTATION_ERROR,
    LOG_FILE_NOT_FOUND,
)

VOCABULARY = frozenset(CLASSES + DOMAIN_TYPES)

_NOT_APPLIED = re.compile(r"[A-Z][A-Za-z]*" + NOT_APPLIED_SUFFIX)

# Names that belong to a class other than the one their spelling suggests, or
# that no family pattern catches. Checked first, by exact name.
_ALIASES: Dict[str, str] = {
    # aiohttp (Unreal Remote Control)
    "ClientConnectorError": CONNECTION_ERROR,
    "ClientConnectorDNSError": CONNECTION_ERROR,
    "ClientOSError": CONNECTION_ERROR,
    "ServerDisconnectedError": CONNECTION_ERROR,
    "IncompleteReadError": CONNECTION_ERROR,
    # Isaac Sim transports
    "EmptyResponse": CONNECTION_ERROR,
    "BridgeCircuitOpenError": CONNECTION_ERROR,
    # A reply the bridge sent after it took the request: not the caller's
    # input, though it is a ValueError.
    "BridgeResponseProtocolError": BRIDGE_PROTOCOL_ERROR,
    # Python
    "BrokenPipeError": CONNECTION_ERROR,
    "PermissionError": PERMISSION_DENIED,
    "NotImplementedError": UNSUPPORTED_OPERATION,
    "FileNotFoundError": NOT_FOUND_ERROR,
    "KeyError": NOT_FOUND_ERROR,
    "LookupError": NOT_FOUND_ERROR,
    "IndexError": NOT_FOUND_ERROR,
    # A module the engine's Python lacks (Isaac Sim 6.0 dropped the
    # omni.isaac.* shims): the engine's, not a name the caller got wrong.
    "ImportError": ENGINE_ERROR,
    "ModuleNotFoundError": ENGINE_ERROR,
    # asyncio's, about a future: not the caller's input, though "Invalid*".
    "InvalidStateError": ENGINE_ERROR,
    # A reply that would not decode: the engine's output, not the caller's.
    "UnicodeDecodeError": ENGINE_ERROR,
    "UnicodeError": ENGINE_ERROR,
}

#: The failure names the khemoo.simul Isaac bridge extension and the Isaac
#: socket client report for a request they refused or could not carry, which
#: are not an exception the engine's code raised (``BridgeResponse.failure``
#: in the bridge, the client's ``ScriptResult.error_name`` markers).
ISAAC_TRANSPORT_FAILURES: Dict[str, str] = {
    # The bridge extension
    "AuthenticationError": AUTHENTICATION_ERROR,
    "ExtensionNotReady": CONNECTION_ERROR,
    "RequestReadTimeout": CONNECTION_ERROR,
    "RequestTimeout": TIMEOUT_ERROR,
    "ResponseTooLarge": PAYLOAD_TOO_LARGE,
    "UnknownAction": UNSUPPORTED_OPERATION,
    "UnsafeExecutionDisabled": SCRIPT_EXECUTION_DISABLED,
    "StageUnavailable": INVALID_STATE,
    "PrimNotFound": NOT_FOUND_ERROR,
    "PrimNotXformable": VALIDATION_ERROR,
    "InvalidRequest": VALIDATION_ERROR,
    "InvalidRootPath": VALIDATION_ERROR,
    "InvalidAspect": VALIDATION_ERROR,
    "InvalidSearchType": VALIDATION_ERROR,
    "InvalidSimulationCommand": VALIDATION_ERROR,
    # The socket client
    "EmptyResponse": CONNECTION_ERROR,
    "ScriptTooLargeForSocket": PAYLOAD_TOO_LARGE,
    "UnknownError": ENGINE_ERROR,
}

# Names that mean a transport failure only when the client says it produced
# them (``ScriptResult.transport_failure``); raised by a script, they are the
# script's: a reply that is not JSON, a script the bridge stopped at its timeout.
_ISAAC_TRANSPORT_MARKERS: Dict[str, str] = {
    "JSONDecodeError": BRIDGE_PROTOCOL_ERROR,
    "ScriptInterrupted": TIMEOUT_ERROR,
}

_FILE_NOT_FOUND_ERRNOS = frozenset({errno.ENOENT})
_PERMISSION_ERRNOS = frozenset({errno.EACCES, errno.EPERM})
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

#: Longest ``error`` (and ``details.errors[].message``) an envelope carries;
#: a message that repeats a long argument is cut, saying how much was left out.
MAX_MESSAGE_CHARS = 2000
# Longest exception name kept in ``details.exception_type``.
_MAX_NAME_CHARS = 128

_EMPTY_MESSAGES = {
    TIMEOUT_ERROR: "The engine did not answer in time; the operation may still be running.",
    CONNECTION_ERROR: "The connection to the engine failed before it answered.",
}
_DEFAULT_MESSAGE = "The operation failed without a message."

# bpy's message for an enum property set to a value it does not have:
#   bpy_struct: item.type = val: enum "BOXX" not found in ('BOX', 'SPHERE')
# Anchored on the literal ``enum "``, with bounded runs, so a long message
# costs one linear scan; the field is looked for only in the few characters
# before the match (``_BPY_FIELD``). A leading optional ``[\w.]*`` group
# backtracked quadratically on a long run of word characters.
_BPY_ENUM = re.compile(
    r'enum\s{1,8}"(?P<given>[^"\n]{0,256})"\s{1,8}not\s{1,8}found\s{1,8}in\s{1,8}'
    r"\((?P<choices>[^)\n]{0,4096})\)"
)
_BPY_FIELD = re.compile(r"(?P<field>[A-Za-z_][\w.]{0,127})\s{0,8}=\s{0,8}\w{1,64}:\s{0,8}$")
_BPY_FIELD_WINDOW = 200
_QUOTED = re.compile(r"'([^'\\\n]{0,256})'")
# A tool's own argument check names the parameter first:
#   stiffness must be >= 0, got -1.0
#   min_position_iterations (5) exceeds max_position_iterations (2)
#   Invalid mode 'Foo'. Must be one of [...]
# Matched at the start only, with bounded runs, so it costs a few characters.
_LEADING_FIELD = re.compile(
    r"(?:Invalid|Unknown|Unsupported)\s(?P<named>[a-z_][a-z0-9_]{0,63})\s['\"(]"
    r"|(?P<field>[a-z_][a-z0-9_]{0,63}(?:\.[a-z_][a-z0-9_]{0,63}){0,4})"
    r"(?:\s\([^)\n]{0,64}\))?"
    r"\s(?:must|should|cannot|can't|is\snot|is\srequired|exceeds|has\sto)\b"
)


def field_named_by(message: str) -> Optional[str]:
    """The parameter a tool's own argument check names at the start of its message, or None.

    Args:
        message: The ValidationError message (``stiffness must be >= 0``).

    Returns:
        The parameter (``stiffness``), or None when the message does not
        start by naming one.
    """
    match = _LEADING_FIELD.match(message)
    if match is None:
        return None
    return match.group("named") or match.group("field")


def is_known(name: Optional[str]) -> bool:
    """Whether ``name`` is in the vocabulary (a class, a domain type or a ``*NotApplied``)."""
    return bool(name) and (name in VOCABULARY or bool(_NOT_APPLIED.fullmatch(str(name))))


def family_of(name: str) -> str:
    """Return the vocabulary type a name outside the vocabulary stands for.

    By exact name first (``PermissionError`` is ``PermissionDenied``,
    ``BridgeResponseProtocolError`` a ``BridgeProtocolError``), then by
    family: a name with ``Timeout`` in it is a ``TimeoutError``; one with
    ``Connection`` or ``BrokenPipe`` a ``ConnectionError``; one with
    ``NotFound`` a ``NotFoundError``; ``ValueError``, ``TypeError``,
    ``JSONDecodeError``, ``Invalid*`` and ``*ValidationError`` a
    ``ValidationError``. Anything else (``OSError``, ``RuntimeError``,
    ``ImportError``) is an ``EngineError``.

    Args:
        name: An exception or error name; a known one is returned unchanged.
    """
    if is_known(name):
        return name
    if name in _ALIASES:
        return _ALIASES[name]
    if "Timeout" in name:
        return TIMEOUT_ERROR
    # Not OSError by name: relayed from the engine (a full disk inside Kit or
    # Blender), it is the engine's failure, not a connection that broke.
    if "Connection" in name or "BrokenPipe" in name:
        return CONNECTION_ERROR
    if "NotFound" in name:
        return NOT_FOUND_ERROR
    if (
        name in ("ValueError", "TypeError", "JSONDecodeError")
        or name.startswith("Invalid")
        or name.endswith("ValidationError")
    ):
        return VALIDATION_ERROR
    return ENGINE_ERROR


def is_mapped(name: str) -> bool:
    """Whether ``name`` is in the vocabulary or a name the module maps on purpose."""
    return (
        is_known(name)
        or name in _ALIASES
        or name in ISAAC_TRANSPORT_FAILURES
        or name in _ISAAC_TRANSPORT_MARKERS
    )


def engine_failure_type(
    name: str, *, transport_failure: bool = False, raised: str = ENGINE_ERROR
) -> str:
    """Return the vocabulary type of a failure an Isaac Sim transport reported.

    The bridge's and the socket client's own failure names
    (``ISAAC_TRANSPORT_FAILURES``) map onto their types. Any other name is an
    exception the engine's code raised, a script inside Kit: a ``TypeError``
    or an ``OSError`` there is the engine's failure, not the caller's input
    or a broken connection, so it is ``raised`` (``EngineError`` for a tool's
    own script, ``ScriptError`` for an agent's), whatever its spelling. Only
    a failure the client marks as its own (``transport_failure``) is read by
    its name: a reply that is not JSON is a ``BridgeProtocolError``, a script
    the bridge stopped at its timeout a ``TimeoutError``.

    Args:
        name: The failure name the transport reported.
        transport_failure: The client produced the failure itself.
        raised: The type of an exception the engine's code raised.
    """
    if name in ISAAC_TRANSPORT_FAILURES:
        return ISAAC_TRANSPORT_FAILURES[name]
    if transport_failure:
        return _ISAAC_TRANSPORT_MARKERS.get(name) or family_of(name)
    return raised


def bounded_message(message: str) -> str:
    """Return ``message`` cut to ``MAX_MESSAGE_CHARS``, saying how much was left out."""
    if len(message) <= MAX_MESSAGE_CHARS:
        return message
    return f"{message[:MAX_MESSAGE_CHARS]}... <{len(message) - MAX_MESSAGE_CHARS} more characters>"


def _bounded_name(name: str) -> str:
    """An exception name as ``details.exception_type`` keeps it."""
    return name if len(name) <= _MAX_NAME_CHARS else f"{name[:_MAX_NAME_CHARS]}..."


def canonical_error_type(name: Optional[str]) -> Optional[str]:
    """Return the vocabulary type for a payload's ``error_type``, or None when it has none."""
    if not name:
        return None
    return family_of(name)


def classify_exception(exc: BaseException) -> str:
    """Return the vocabulary type of an exception that escaped a tool call.

    A declared ``error_type`` (a structured exception, an error the Blender
    add-on relays) is used when it is in the vocabulary and mapped by its
    family otherwise. An undeclared exception is classified by what it is: a
    timeout, a connection failure (by class, by library name, or by a socket
    ``errno``), a missing file or a refused permission by ``errno``, then by
    the first class in its MRO with a known name or family.

    Args:
        exc: The exception.

    Returns:
        A name in the vocabulary.
    """
    declared = getattr(exc, "error_type", None)
    if isinstance(declared, str) and declared:
        return family_of(declared)
    if isinstance(exc, TimeoutError):
        return TIMEOUT_ERROR
    mro = [klass.__name__ for klass in type(exc).__mro__ if klass not in (BaseException, Exception, object)]
    for name in mro:
        if name in _ALIASES:
            return _ALIASES[name]
    if isinstance(exc, ConnectionError):
        return CONNECTION_ERROR
    if type(exc) is OSError and exc.errno is None:
        # Raised here, not relayed: asyncio's "Multiple exceptions: ..." when
        # no address of a host would connect.
        return CONNECTION_ERROR
    if isinstance(exc, OSError):
        if exc.errno in _CONNECTION_ERRNOS:
            return CONNECTION_ERROR
        if exc.errno in _FILE_NOT_FOUND_ERRNOS:
            return NOT_FOUND_ERROR
        if exc.errno in _PERMISSION_ERRNOS:
            return PERMISSION_DENIED
        # A file or device error (a full disk, a read-only file system): the
        # engine could not do it, not a broken connection.
        return ENGINE_ERROR
    for name in mro:
        family = family_of(name)
        if family != ENGINE_ERROR:
            return family
    return ENGINE_ERROR


def exception_fields(exc: BaseException) -> Dict[str, Any]:
    """Return the ``error_type`` and ``details`` an envelope reports for ``exc``.

    Args:
        exc: The exception.

    Returns:
        ``{"error_type": ..., "details": ...}`` for an ``ErrorResponse``:
        the vocabulary type, and the exception's own ``details`` with its
        name under ``exception_type`` when that differs from the type
        (None when there is nothing to report).
    """
    error_type = classify_exception(exc)
    details = getattr(exc, "details", None)
    details = dict(details) if isinstance(details, dict) else {}
    original = exception_type_name(exc)
    if original != error_type:
        details.setdefault("exception_type", _bounded_name(original))
    return {"error_type": error_type, "details": details or None}


def errors_from_message(message: str) -> List[Dict[str, Any]]:
    """Describe a ValidationError that has only a message as one ``details.errors`` entry.

    The entry's ``field`` is the parameter the message starts by naming, as a
    tool's own argument check words it (``stiffness must be >= 0``,
    ``Invalid mode 'Foo'``), else None. A bpy enum refusal
    (``enum "BOXX" not found in ('BOX', 'SPHERE')``) also gets its ``field``
    when the message names it, the ``constraint`` with the choices, the
    ``input`` given and the closest choice as ``suggestion``.

    Args:
        message: The error message; only its first ``MAX_MESSAGE_CHARS`` are
            read, and kept.
    """
    message = bounded_message(message)
    entry: Dict[str, Any] = {"field": field_named_by(message), "message": message, "type": "value_error"}
    match = _BPY_ENUM.search(message, 0, MAX_MESSAGE_CHARS)
    if match is not None:
        choices = _QUOTED.findall(match.group("choices"))
        given = match.group("given")
        before = message[max(0, match.start() - _BPY_FIELD_WINDOW):match.start()]
        named = _BPY_FIELD.search(before)
        field = named.group("field") if named is not None else None
        entry.update(
            field=field.rsplit(".", 1)[-1] if field else None,
            type="enum",
            constraint={"expected": choices},
            input=given,
        )
        closest = difflib.get_close_matches(given, choices, n=1, cutoff=0.6)
        if closest:
            entry["suggestion"] = closest[0]
    return [entry]


def normalize_error_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Give a failed payload a type in the vocabulary and a message (in place).

    A name outside the vocabulary becomes the type of its family, with the
    original kept as ``details.exception_type``. ``error`` and each
    ``details.errors`` message are cut to ``MAX_MESSAGE_CHARS``. A failure without any
    ``error_type`` gets one: ``ConnectionError`` when the payload says the
    engine is not reachable (``reachable`` or ``connected`` false),
    ``EngineError`` otherwise. A ``ValidationError`` without
    ``details.errors`` gets its message as the one error. A failure with an
    empty message gets one: the ``*_error`` keys of a partial failure, or a
    sentence saying what happened.

    Args:
        payload: A tool payload whose ``success`` is False.

    Returns:
        The same dict.
    """
    raw = payload.get("error_type")
    current = raw if isinstance(raw, str) and raw else None
    if current is None:
        unreachable = payload.get("reachable") is False or payload.get("connected") is False
        payload["error_type"] = CONNECTION_ERROR if unreachable else ENGINE_ERROR
    elif not is_known(current):
        details = payload.get("details")
        details = dict(details) if isinstance(details, dict) else {}
        details.setdefault("exception_type", _bounded_name(current))
        payload["details"] = details
        payload["error_type"] = family_of(current)
    if payload["error_type"] == VALIDATION_ERROR:
        # Every ValidationError lists its failures, whichever layer raised it.
        details = payload.get("details")
        details = dict(details) if isinstance(details, dict) else {}
        if not isinstance(details.get("errors"), list):
            details["errors"] = errors_from_message(str(payload.get("error") or ""))
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
    if isinstance(payload["error"], str):
        payload["error"] = bounded_message(payload["error"])
    details = payload.get("details")
    if isinstance(details, dict) and isinstance(details.get("errors"), list):
        for entry in details["errors"]:
            if isinstance(entry, dict) and isinstance(entry.get("message"), str):
                entry["message"] = bounded_message(entry["message"])
    return payload
