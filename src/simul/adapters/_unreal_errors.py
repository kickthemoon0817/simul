"""The error types Unreal tools report, and the Remote Control answers they come from.

Every failure an Unreal tool reports carries one ``error_type`` from a small set
shared with the other backends, so an agent can decide what to do from the type
alone:

``ConnectionError``
    Nothing answered at the Remote Control endpoint: no editor is listening,
    or the editor closed the connection without answering (it crashed or is
    shutting down).
``TimeoutError``
    The editor did not answer in time. A mutation or script may still be
    running in the editor, and Remote Control cannot cancel it, so check the
    editor's state before retrying.
``EngineError``
    The editor took the request and failed it (an HTTP 5xx answer).
``ValidationError``
    Remote Control refused the request as made (HTTP 400): a read-only or
    protected property, a value of the wrong type, a malformed body.
``NotFoundError``
    The object, property or function named does not exist: an HTTP 404, an
    HTTP 400 that says so, or a tool script's ``SimulNotFound``.
``InstanceBusy``
    The editor answered but cannot serve the request now (it is saving or
    collecting garbage).
``SandboxError``
    The path policy refused a path on this machine (``simul.utils.paths``).

``AuthenticationError`` (HTTP 401: Remote Control refused this client, for its
passphrase or its address) and the domain types (``PIEActive`` and the like)
keep their names. The strings are the cross-backend set; when a shared module
defines them, these constants should import from it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

CONNECTION_ERROR = "ConnectionError"
TIMEOUT_ERROR = "TimeoutError"
ENGINE_ERROR = "EngineError"
VALIDATION_ERROR = "ValidationError"
NOT_FOUND_ERROR = "NotFoundError"
INSTANCE_BUSY = "InstanceBusy"
SANDBOX_ERROR = "SandboxError"
AUTHENTICATION_ERROR = "AuthenticationError"


class UnrealError(Exception):
    """A failed Unreal request, named by its ``error_type``.

    The MCP envelope and the CLI report an exception by its ``error_type``
    attribute (``simul.utils.errors.exception_type_name``) and attach
    ``details``, so raising one of these is how the transport reports a
    failure in the shared taxonomy.

    Attributes:
        error_type: One of the module's error types.
        status: The HTTP status Remote Control answered with, or None when it
            did not answer.
        details: Context for the caller (the Remote Control route, its
            message), or None.
    """

    def __init__(
        self,
        message: str,
        error_type: str,
        *,
        status: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status = status
        self.details = details or None


class UnrealTimeoutError(UnrealError, TimeoutError):
    """The editor did not answer in time; still a ``TimeoutError`` to ``except`` clauses."""


class UnrealValidationError(UnrealError, ValueError):
    """A request refused before anything was sent; still a ``ValueError`` to ``except`` clauses."""

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, VALIDATION_ERROR, details=details)


class UnrealConnectionError(UnrealError, ConnectionError):
    """Nothing answered at the endpoint; still a ``ConnectionError`` to ``except`` clauses.

    Attributes:
        os_error: The socket error behind it, when there was one, as aiohttp's
            ``ClientConnectorError.os_error`` keeps it.
    """

    os_error: Optional[OSError] = None


# Remote Control builds its error body with an unescaped printf,
# ``{ "errorMessage": "%s" }`` (WebRemoteControlInternalUtils::
# CreateUTF8ErrorMessage, UE 5.7), so a message with a newline or a quote is
# not valid JSON; this takes whatever sits between the quotes.
_ERROR_MESSAGE = re.compile(r'"errorMessage"\s*:\s*"(.*)"\s*}\s*$', re.DOTALL)

# Remote Control's wording for a name that resolves to nothing (UE 5.7
# RemoteControlModule::ResolveObject / ResolveObjectProperty, and the call and
# preset routes): "Object: <path> does not exist ...", "Object property: <name>
# could not be resolved on object: <path>", "Unable to find the object.",
# "The object or function was not found.", "Unable to resolve the preset.".
_NOT_FOUND_WORDING = re.compile(
    r"does not exist|could not be resolved|not found|unable to find|unable to resolve",
    re.IGNORECASE,
)

# "Can't resolve object: <path> while saving or garbage collecting."
_BUSY_WORDING = re.compile(r"while saving or garbage collecting", re.IGNORECASE)


def remote_control_message(body: str) -> str:
    """Remote Control's ``errorMessage`` in an error body, or the body itself.

    Args:
        body: The response body as text.

    Returns:
        The message, stripped; the whole body when it carries none; empty when
        the body is empty.
    """
    text = body.strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        message = parsed.get("errorMessage")
        if isinstance(message, str):
            return message.strip()
        return text
    match = _ERROR_MESSAGE.search(text)
    return match.group(1).strip() if match else text


def remote_control_error_type(status: int, message: str) -> str:
    """The error type of a Remote Control answer with an HTTP error status.

    Args:
        status: The HTTP status, 400 or above.
        message: Remote Control's message, possibly empty.

    Returns:
        ``AuthenticationError`` for 401, ``NotFoundError`` for 404 or a 400
        whose message says the name resolved to nothing, ``InstanceBusy`` when
        the editor was saving or collecting garbage, ``EngineError`` for 5xx,
        and ``ValidationError`` for any other refusal.
    """
    if status == 401:
        return AUTHENTICATION_ERROR
    if _BUSY_WORDING.search(message):
        return INSTANCE_BUSY
    if status == 404 or (status < 500 and _NOT_FOUND_WORDING.search(message)):
        return NOT_FOUND_ERROR
    if status >= 500:
        return ENGINE_ERROR
    return VALIDATION_ERROR


def remote_control_error(status: int, body: str, method: str, path: str) -> UnrealError:
    """The exception for a Remote Control answer with an HTTP error status.

    Args:
        status: The HTTP status.
        body: The response body as text.
        method: The request's HTTP method.
        path: The request's route.

    Returns:
        An ``UnrealError`` whose message carries Remote Control's own (which
        aiohttp's ``raise_for_status`` dropped), with the route and the status
        in ``details``.
    """
    message = remote_control_message(body)
    error_type = remote_control_error_type(status, message)
    text = f"Remote Control refused {method} {path} (HTTP {status})"
    return UnrealError(
        f"{text}: {message}" if message else text,
        error_type,
        status=status,
        details={"status": status, "route": path, "remote_control_message": message or None},
    )


#: The exception class a tool script raises for a name that resolves to
#: nothing; ``_execute_json_script`` reports it as ``NotFoundError``. The
#: editor-side helpers (``_unreal_scripts.ACTOR_HELPERS``) define and raise it.
SCRIPT_NOT_FOUND_CLASS = "SimulNotFound"

# The traceback's last line for it: "SimulNotFound: <message>", qualified by
# its module when the script ran in one.
_SCRIPT_NOT_FOUND = re.compile(
    r"^(?:[\w.]+\.)?" + SCRIPT_NOT_FOUND_CLASS + r": (?P<message>.+?)\s*$", re.MULTILINE
)


def script_not_found(failure: str) -> Optional[str]:
    """The message of a ``SimulNotFound`` a tool script raised, or None.

    Args:
        failure: The text Unreal returned for a failed script (its traceback).

    Returns:
        The exception's message when the script ended with an uncaught
        ``SimulNotFound``, otherwise None.
    """
    matches = list(_SCRIPT_NOT_FOUND.finditer(failure))
    return matches[-1].group("message") if matches else None
