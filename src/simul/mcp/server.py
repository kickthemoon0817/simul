"""
Simul 3D MCP Server implementation.

This module provides the main MCP server class with tool registry,
connection management, and 3D simulation/DCC integration based on FastMCP.
"""

import asyncio
import base64
import binascii
import difflib
import hashlib
import inspect
import json
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

import anyio
from fastmcp import FastMCP
from fastmcp.server.context import _current_context
from fastmcp.tools import ToolResult
from fastmcp.utilities.tasks import TaskConfig as _TaskConfig
from mcp.types import ImageContent, TextContent, ToolAnnotations
from mcp.types.version import MODERN_PROTOCOL_VERSIONS
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from .. import __version__ as _source_version
from ..adapters import IsaacRuntimeAdapter, IsaacSocketClient
from ..adapters.base import BackendAdapter
from ..config import Settings, get_settings
from ..logging import LoggerMixin, current_request_id, get_logger, note_call_outcome
from ..resources import find_checkout_root, resource
from ..utils.errors import exception_message, exception_type_name
from ..utils.paths import PathPolicy, SandboxDenied, sandbox_error
from ..utils.timing import RateLimiter
from . import error_types
from .backends import ALL_BACKEND_NAMES, BACKENDS, NOT_SELECTED, backend_skip_reason
from .error_types import normalize_error_payload
from .registration import register_stats_tools
from .registration._helpers import apply_success_from_error
from .result_budget import encode_result_budget, tool_result
from .schemas.common import ErrorResponse
from .session_manager import (
    CLAIM_TTL_SECONDS,
    ClaimConflict,
    ClaimStoreError,
    ClaimView,
    SessionManager,
    claim_is_orphaned,
)
from ..utils.discovery import DiscoveryDir
from .tools.isaac_tools import IsaacTools
from .usage_tracker import ToolUsageTracker

logger = get_logger(__name__)

# sizeof(sockaddr_un.sun_path), terminating NUL included: 104 on macOS and the
# BSDs, 108 on Linux.
_UNIX_SOCKET_PATH_MAX = 108 if sys.platform.startswith("linux") else 104

# The version MCP clients see in serverInfo. simul.__version__ is the one
# tests/test_version_lockstep.py ties to pyproject.toml; the installed
# distribution's metadata is stale in an editable install (it said 0.1.6 while
# `simul version` said 0.1.11).
_PACKAGE_VERSION: str = _source_version


def build_instructions(
    backends: Iterable[str], tool_backends: Optional[Mapping[str, str]] = None
) -> str:
    """Write the server's MCP instructions for the backends whose tools it registers.

    A backend that registers nothing is not mentioned, and neither are the
    SimReady helpers unless one of them is registered, so the instructions
    never point an agent at a tool it cannot see. The text for each backend
    comes from its ``BackendSpec``: the summary sentence, its guidance
    paragraphs and its routing rule.

    Args:
        backends: Registry names of the backends whose tools are registered.
        tool_backends: Each registered tool's name mapped to the backend that
            registered it. The SimReady routing line names the backends that
            registered a tool containing ``simready`` and is left out when
            none did, and a backend's script guidance is stated only when its
            script tool registered; omitting this argument means neither did.

    Returns:
        The instructions text.
    """
    names = set(backends)
    specs = [spec for spec in BACKENDS if spec.name in names]
    simready_owners = {
        backend for tool, backend in (tool_backends or {}).items() if "simready" in tool
    }
    simready_labels = [spec.label for spec in specs if spec.name in simready_owners]
    intro = "Simul provides tools for interacting with 3D simulation and DCC applications."
    paragraphs = [
        " ".join([intro, *(spec.summary for spec in specs)]),
        "TOOL SELECTION — prefer a granular tool when one covers the operation; they are "
        "cheaper and return structured results. Every tool parameter's units and "
        "conventions are in its schema description.",
    ]
    registered_tools = set(tool_backends or {})
    paragraphs += [
        _guidance(spec, registered_tools) for spec in specs if spec.guidance
    ]
    routing = [
        "ROUTING — the backend name appears somewhere inside the tool name, usually as an "
        "infix (capture_<backend>_viewport):"
    ]
    routing += [f"  {spec.routing_rule}" for spec in specs if spec.name != "usd"]
    if simready_labels:
        routing.append(
            "  Tools containing 'simready' (SimReady asset helpers) belong to "
            f"{' or '.join(simready_labels)}; each one's description names the runtime "
            "it targets."
        )
    routing += [f"  {spec.routing_rule}" for spec in specs if spec.name == "usd"]
    routing.append("  get_tool_usage_stats is server metadata, not a backend.")
    paragraphs.append("\n".join(routing))
    return "\n\n".join(paragraphs)


def _guidance(spec: Any, registered_tools: Set[str]) -> str:
    """Return a backend's guidance, with its script sentences only when the script tool registered.

    The script guidance joins the end of the guidance's first line, where it
    always stood, so the text is unchanged when scripting is allowed.
    """
    if not spec.script_guidance or spec.script_tool not in registered_tools:
        return str(spec.guidance)
    head, separator, tail = spec.guidance.partition("\n")
    return f"{head} {spec.script_guidance}{separator}{tail}"


_FASTMCP_SUPPORTS_INSTRUCTIONS: bool = (
    "instructions" in inspect.signature(FastMCP).parameters
)

# The background task config FastMCP takes; None switches task support off.
TaskConfig: Optional[Type[_TaskConfig]] = _TaskConfig

# Tool key of the bucket that caps one agent's calls across every tool.
_GLOBAL_RATE_BUCKET: str = "*"

# The name a call to a tool that is not registered is recorded and rate
# limited under; the name it gave is kept in the record's parameters.
_UNKNOWN_TOOL: str = "<unknown tool>"

# Tools the rate limit never refuses: releasing a claim must work for an agent
# that has spent its budget, or the claim would outlive the agent's patience.
_RATE_LIMIT_EXEMPT = frozenset({"release_isaac_instance"})

# Transports ``SimulMCPServer.run`` accepts. ``http`` is streamable HTTP, the
# current MCP network transport; ``sse`` is kept for clients that still speak it.
NETWORK_TRANSPORTS: Tuple[str, ...] = ("http", "sse")
TRANSPORTS: Tuple[str, ...] = ("stdio", *NETWORK_TRANSPORTS)

# The network transports as FastMCP's ``Context.transport`` names them.
_CONTEXT_NETWORK_TRANSPORTS: Tuple[str, ...] = ("streamable-http", "sse")

# The agent a request without a client identity acts as (see
# ``SimulMCPServer._request_is_anonymous``). Nothing tells such clients apart,
# so they share this agent's rate-limit buckets and keep no per-client state.
_ANONYMOUS_AGENT: str = "anonymous"

# The HTTP header an anonymous client names itself in, for the logs only.
_CLIENT_LABEL_HEADER: str = "x-simul-agent"

# The attribute a cancelled Isaac call's CancelledError carries its
# CancelledOutcome under, from _exec_isaac to _run_envelope's record.
_DRAINED_OUTCOME = "_simul_drained_outcome"


@dataclass(frozen=True)
class CancelledOutcome:
    """What the engine did with a call the client cancelled.

    Attributes:
        executed: True when the engine ran the operation, ``"unknown"`` when
            nothing says whether it did.
        payload: The engine's answer, normalised like any call's, when it gave one.
    """

    executed: Union[bool, str]
    payload: Optional[Dict[str, Any]]


# How long a cancelled Isaac call waits for the bridge to accept its interrupt.
_INTERRUPT_TIMEOUT_SECONDS: float = 5.0

# Rate-limit buckets kept before idle ones are swept. A bucket that has
# refilled to its burst is indistinguishable from a new one, so dropping it
# changes nothing; the sweep only bounds the table for identities that never
# end a session (authenticated sessionless clients).
_RATE_LIMITER_SWEEP_SIZE: int = 512


def _ceiling_key(agent: str) -> str:
    """The agent whose ceiling across every tool a call spends: an anonymous peer's host."""
    prefix = f"{_ANONYMOUS_AGENT}@"
    if agent.startswith(prefix):
        host, _, _port = agent[len(prefix):].rpartition(":")
        if host:
            return f"{prefix}{host}"
    return agent


def default_agent_label(session_id: str) -> str:
    """Return the published agent label for a session that names none.

    A one-way digest of the transport session id: the label is shown to every
    client (``list_isaac_instances`` lists claim holders), while over HTTP the
    session id itself is what a client presents as ``Mcp-Session-Id`` to act
    in that session, so it must never be the label.
    """
    digest = hashlib.sha256(f"agent-label:{session_id}".encode("utf-8")).hexdigest()
    return f"session-{digest[:12]}"


def _request_principal() -> Optional[str]:
    """Return the authenticated principal of the current request, or None.

    FastMCP's own identity for per-user state on the sessionless protocol: the
    access token's client, issuer and subject. Imported here because only a
    sessionless request needs it.
    """
    from fastmcp.server.sessions import current_principal

    return current_principal()


# Leading bytes that identify an encoded image, whatever the payload claims.
_IMAGE_SIGNATURES: Tuple[Tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF8", "gif"),
)


def _sniff_image_format(image_base64: str) -> Optional[str]:
    """Identify a base64 image from its magic bytes.

    A capture's bytes are authoritative: a payload without a ``format`` key
    (or with a stale one) must not be labelled with the wrong MIME type.

    Args:
        image_base64: Base64-encoded image data.

    Returns:
        ``png``, ``jpeg``, ``gif`` or ``webp``, or None when unrecognised.
    """
    try:
        head = base64.b64decode(image_base64[:24], validate=False)
    except (ValueError, binascii.Error):
        return None
    for signature, name in _IMAGE_SIGNATURES:
        if head.startswith(signature):
            return name
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


# Longest argument string a ValidationError echoes in details.input; a script
# or a document is replaced by its size.
_ECHO_MAX_CHARS: int = 256
# Most list items or mapping keys echoed from one list or mapping, and most
# scalars echoed from one argument; the rest is counted, not shown.
_ECHO_MAX_ITEMS: int = 16
_ECHO_MAX_VALUES: int = 64
# Nesting deeper than this is echoed by its type and size.
_ECHO_MAX_DEPTH: int = 4

# A name is secret when one of its words is one of these. Words split at
# ``_``, ``-``, ``.``, ``/``, spaces and camelCase humps, so ``access_key``,
# ``privateKey``, ``X-Api-Key`` and ``/persistent/auth_token`` match and
# ``author`` or ``keyframe`` do not.
_SECRET_WORDS = frozenset(
    {
        "token", "tokens", "password", "passwords", "passphrase", "passwd", "pwd",
        "secret", "secrets", "key", "apikey", "auth", "authorization", "credential",
        "credentials", "cookie", "cookies", "bearer", "jwt", "pin", "otp", "accesstoken",
        "authtoken", "accesskey", "privatekey", "secretkey",
    }
)
_NAME_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_MASKED = "***"
# A quoted choice in pydantic's "Input should be 'a', 'b' or 'c'".
_QUOTED = re.compile(r"'([^'\\\n]{0,256})'")
# Longest text a regex or a closest-match search in the envelope reads: the
# caller chooses its length, and the envelope runs on the event loop.
_SCAN_MAX_CHARS: int = 4096
# The context keys of a pydantic error that state the constraint that failed
# (not ``actual_length``, ``class_name`` or ``error``, which describe the input
# or the failure).
_CONSTRAINT_KEYS = frozenset(
    {
        "gt", "ge", "lt", "le", "multiple_of", "min_length", "max_length", "expected",
        "pattern", "max_digits", "decimal_places", "allowed_schemes",
    }
)


def _names_secret(name: str) -> bool:
    """Whether an argument name, mapping key or settings path names a secret."""
    return any(
        word.lower() in _SECRET_WORDS for word in _NAME_WORD.findall(name[:_SCAN_MAX_CHARS])
    )


def _jsonable(value: Any) -> Any:
    """Return ``value`` if JSON can carry it, else its text."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _echo_name(name: str, index: int) -> str:
    """A key or argument name as an echo shows it: a long one by its size, numbered to stay apart."""
    return name if len(name) <= _ECHO_MAX_CHARS else f"<name {index} of {len(name)} characters>"


def _echo_value(field: Optional[str], value: Any) -> Any:
    """The value a ValidationError shows for ``field``, bounded and with its secrets masked.

    A value is masked (``***``) when ``field``, a dotted path, names a secret,
    and so is every value nested under a mapping key that does (a key such as
    ``api_key`` or a settings path such as ``/persistent/auth_token``). A
    string longer than ``_ECHO_MAX_CHARS`` is shown by its size; a list or
    mapping shows its first ``_ECHO_MAX_ITEMS`` entries and counts the rest,
    and the whole value at most ``_ECHO_MAX_VALUES`` scalars.
    """
    budget = [_ECHO_MAX_VALUES]

    def echo(name: Optional[str], item: Any, depth: int) -> Any:
        if name is not None and _names_secret(name):
            return _MASKED
        if isinstance(item, (dict, list, tuple)) and depth >= _ECHO_MAX_DEPTH:
            return f"<{type(item).__name__} of {len(item)}>"
        if isinstance(item, dict):
            entries = list(item.items())
            shown: Dict[str, Any] = {}
            for index, (key, nested) in enumerate(entries[:_ECHO_MAX_ITEMS]):
                if budget[0] <= 0:
                    break
                name = str(key)
                shown[_echo_name(name, index)] = echo(name, nested, depth + 1)
            if len(shown) < len(entries):
                shown["..."] = f"<{len(entries) - len(shown)} more keys, {len(entries)} in all>"
            return shown
        if isinstance(item, (list, tuple)):
            listed: List[Any] = []
            for nested in item[:_ECHO_MAX_ITEMS]:
                if budget[0] <= 0:
                    break
                listed.append(echo(None, nested, depth + 1))
            if len(listed) < len(item):
                listed.append(f"<{len(item) - len(listed)} more items, {len(item)} in all>")
            return listed
        budget[0] -= 1
        if item is not None and not isinstance(item, (bool, int, float, str)):
            item = str(item)
        if isinstance(item, str) and len(item) > _ECHO_MAX_CHARS:
            return f"<{len(item)} characters>"
        return item

    return echo(field, value, 0)


def _suggestion(
    error_type: str, given: Any, constraint: Mapping[str, Any], known_fields: Iterable[str]
) -> Optional[str]:
    """Return the closest valid choice or argument name for a misspelt one, or None."""
    if isinstance(given, str) and len(given) > _ECHO_MAX_CHARS:
        # Nothing that long is a misspelling of a name.
        return None
    if error_type in ("literal_error", "enum") and isinstance(given, str):
        choices = _QUOTED.findall(str(constraint.get("expected", ""))[:_SCAN_MAX_CHARS])
    elif error_type in ("unexpected_keyword_argument", "extra_forbidden") and isinstance(given, str):
        choices = list(known_fields)
    else:
        return None
    matches = difflib.get_close_matches(given, choices, n=1, cutoff=0.6)
    return matches[0] if matches else None


def _validation_errors(
    exc: PydanticValidationError, known_fields: Iterable[str] = ()
) -> List[Dict[str, Any]]:
    """Describe each failed field of a pydantic error.

    Each entry has ``field`` (the dotted path, a nested model's field
    included), ``message`` and ``type``, plus when pydantic reports them the
    ``constraint`` that failed (``{"ge": 1}``, ``{"expected": "'a' or 'b'"}``),
    the ``input`` given (masked for a secret argument, a long string by its
    size) and a ``suggestion``: the closest valid choice, or the closest
    argument name for an unexpected one.

    Args:
        exc: The pydantic error.
        known_fields: The tool's argument names, for suggesting a misspelt one.
    """
    known = list(known_fields)
    errors: List[Dict[str, Any]] = []
    for error in exc.errors(include_url=False):
        location = [str(part) for part in error.get("loc", ())]
        dotted = ".".join(location) or None
        field = dotted
        if dotted is not None and len(dotted) > _ECHO_MAX_CHARS:
            field = f"<{len(dotted)} characters>"
        error_type = str(error.get("type", "value_error"))
        message = str(error.get("msg", ""))
        # pydantic prefixes a validator's own message with "Value error, ".
        for prefix in ("Value error, ", "Assertion failed, "):
            if message.startswith(prefix):
                message = message[len(prefix):]
        entry: Dict[str, Any] = {"field": field, "message": message, "type": error_type}
        context = error.get("ctx") or {}
        constraint = {
            key: _jsonable(value) for key, value in context.items() if key in _CONSTRAINT_KEYS
        }
        if constraint:
            entry["constraint"] = constraint
        if "input" in error and error_type not in ("missing", "missing_argument"):
            given = error["input"]
            if error_type in ("unexpected_keyword_argument", "extra_forbidden"):
                # The name is what was wrong; its value is echoed in details.input.
                given_name = location[-1] if location else None
                suggestion = _suggestion(error_type, given_name, constraint, known)
            else:
                entry["input"] = _echo_value(dotted, given)
                suggestion = _suggestion(error_type, given, constraint, known)
            if suggestion is not None:
                entry["suggestion"] = suggestion
        errors.append(entry)
    return errors


def _summarize_validation_errors(errors: List[Dict[str, Any]]) -> str:
    """Join per-field errors into one message: ``field: message; ...``.

    A message that already starts by naming its field (a tool's own check:
    ``stiffness must be >= 0``) is not prefixed with it again.
    """
    return "; ".join(
        f"{error['field']}: {error['message']}"
        if error.get("field") and error_types.field_named_by(str(error["message"])) != error["field"]
        else str(error["message"])
        for error in errors
    )


def _echo_arguments(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the arguments a ValidationError echoes, each bounded with its secrets masked.

    At most ``_ECHO_MAX_ITEMS`` * 2 arguments are shown; a call carrying more
    (unexpected ones) has the rest counted.
    """
    entries = list(arguments.items())
    limit = _ECHO_MAX_ITEMS * 2
    echoed = {
        _echo_name(str(key), index): _echo_value(str(key), value)
        for index, (key, value) in enumerate(entries[:limit])
    }
    if len(entries) > limit:
        echoed["..."] = f"<{len(entries) - limit} more arguments, {len(entries)} in all>"
    return echoed


def validation_error_payload(
    exc: BaseException,
    arguments: Mapping[str, Any],
    *,
    known_fields: Iterable[str] = (),
) -> Dict[str, Any]:
    """Build the one ValidationError envelope every layer returns for invalid input.

    The MCP schema (a wrong type, a missing or unexpected argument, a value
    outside its bounds), a tool's request model and its own argument checks
    all report the same shape: ``error`` joins ``field: message`` for each
    failure, ``details.errors`` lists them, ``details.input`` echoes the
    arguments (bounded, secrets masked: ``_echo_value``).

    Args:
        exc: A pydantic ValidationError, or the exception an argument check
            raised with its message as written.
        arguments: The arguments the call received.
        known_fields: The tool's argument names, for suggesting a misspelt one.

    Returns:
        The error envelope.
    """
    if isinstance(exc, PydanticValidationError):
        errors = _validation_errors(exc, known_fields)
    else:
        errors = error_types.errors_from_message(exception_message(exc))
    return ErrorResponse(
        error=_summarize_validation_errors(errors),
        error_type=error_types.VALIDATION_ERROR,
        details={
            "errors": errors,
            "input": _echo_arguments(arguments),
        },
    ).model_dump()


def build_error_envelope_middleware(server: "SimulMCPServer") -> Any:
    """Build the middleware that runs FastMCP's own call refusals through the shared envelope.

    FastMCP checks a call against the tool's input schema before the tool
    runs, and answered a mismatch (a wrong type, a missing or unexpected
    argument, a value out of bounds or not among the choices, a nested
    model's field) with pydantic's raw text, and an unknown tool with plain
    text, while every other failure is a JSON envelope. This catches both
    where FastMCP raises them (``FunctionTool._run``, inside the middleware
    chain) and answers with the envelope every other failure gets, sent with
    ``isError: true``: ``ValidationError`` with each failing field's path,
    constraint, given value (secrets masked) and closest valid choice in
    ``details.errors``; ``ScriptExecutionDisabled`` for an arbitrary-code
    tool that ``security.allow_script_execution`` keeps off the surface; and
    ``NotFoundError`` for any other tool that is not registered. Every refusal
    goes through the envelope (``_exec_payload``), so it spends a rate-limit
    token and is recorded in the usage log like any other failed call; an
    unknown name is recorded and limited under ``_UNKNOWN_TOOL``, so made-up
    names cannot grow the usage records or the rate-limit buckets.

    Args:
        server: The server whose envelope the refusals go through.
    """
    from fastmcp.exceptions import NotFoundError as FastMCPNotFoundError
    from fastmcp.exceptions import ValidationError as FastMCPValidationError
    from fastmcp.server.middleware import Middleware

    class ErrorEnvelopeMiddleware(Middleware):
        """Answer schema-level refusals with the shared error envelope."""

        async def on_call_tool(self, context: Any, call_next: Any) -> Any:
            message = context.message
            name = str(getattr(message, "name", ""))
            arguments = dict(getattr(message, "arguments", None) or {})
            try:
                return await call_next(context)
            except (FastMCPValidationError, PydanticValidationError) as exc:
                cause = exc if isinstance(exc, PydanticValidationError) else exc.__cause__
                payload = validation_error_payload(
                    cause if isinstance(cause, PydanticValidationError) else exc,
                    arguments,
                    known_fields=await server._tool_parameters(name),
                )
                return await server._exec_payload(name, payload)
            except FastMCPNotFoundError as exc:
                if name in server._hidden_script_tools:
                    return await server._exec_payload(
                        name, server._script_execution_disabled(name)
                    )
                payload = ErrorResponse(
                    error=exception_message(exc),
                    error_type=error_types.NOT_FOUND_ERROR,
                    details={"tool": _echo_value(None, name) or None},
                ).model_dump()
                return await server._exec_payload(
                    _UNKNOWN_TOOL, payload, params={"name": _echo_value(None, name)}
                )

    return ErrorEnvelopeMiddleware()


@dataclass
class IsaacSessionBinding:
    """Per-session binding to one Isaac Sim instance."""

    binding_id: str
    session_id: str
    instance_name: str
    agent_id: str
    port: int
    purpose: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)


@dataclass
class IsaacSessionRoute:
    """Per-session routing and binding state for Isaac Sim access."""

    session_id: str
    active_instance: str
    bindings: Dict[str, IsaacSessionBinding] = field(default_factory=dict)


@dataclass(frozen=True)
class Refusal:
    """The verdict of an envelope stage that a call must not run.

    Attributes:
        payload: The error envelope the caller receives instead of a result.
        usage_error: The short reason the usage log records (``rate_limited``,
            ``instance_claimed``, ``instance_busy``), so refused calls can be
            told apart from calls that ran and failed.
    """

    payload: Dict[str, Any]
    usage_error: str


# A backend's admission hook. The envelope enters it after the rate limit with
# the tool name; it yields a Refusal to stop the call or None to admit it, and
# stays entered until the admitted call has finished, so it can hold a resource
# (Isaac holds the target instance's lock) for exactly that long.
Admission = Callable[[str], AsyncContextManager[Optional[Refusal]]]

# What a USD, Blender or Unreal tool runs inside its session: called with the
# open session, plus the validated request when the tool declares a request
# model; returns the payload, directly or as an awaitable.
BackendCall = Callable[..., Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]

# Builds what a call refused by request validation echoes back as
# ``details.input`` from the tool arguments, for a tool whose arguments are too
# large to echo whole (a script).
InputEcho = Callable[[Mapping[str, Any]], Dict[str, Any]]


@asynccontextmanager
async def _admit_every_call(tool_name: str) -> AsyncIterator[Optional[Refusal]]:
    """The admission hook of a backend with nothing to gate."""
    yield None


class SimulMCPServer(LoggerMixin):
    """
    Simul 3D MCP Server.

    Unified MCP server for 3D simulation and DCC tools. Every backend in
    ``simul.mcp.backends.BACKENDS`` gets its adapter built and its tools
    registered here; Isaac Sim additionally has multi-instance discovery and
    per-session instance routing.
    """

    # All recognised backend names for --backends validation.
    ALL_BACKENDS: frozenset[str] = ALL_BACKEND_NAMES

    def __init__(
        self,
        settings: Optional[Settings] = None,
        backends: Optional[Set[str]] = None,
        require_available: bool = True,
    ):
        """
        Initialize Simul 3D MCP Server.

        Args:
            settings: Configuration settings
            backends: Set of backend names to register MCP tools for.
                      ``None`` (default) registers all available backends,
                      Isaac Sim only where it runs (``isaac_sim.enabled``
                      overrides); naming ``isaac`` registers it on any OS.
                      Valid names: ``isaac``, ``unreal``, ``usd``, ``blender``.
            require_available: When False, register an enabled backend's
                tools even if its runtime is missing here. Only for listing
                the tool surface (``simul tools``); such tools fail on
                every call.
        """
        self.settings = settings or get_settings()
        self._backends = backends  # None means "all available"
        self._require_available = require_available
        self._path_policy = PathPolicy.from_settings(
            self.settings, project_root=find_checkout_root()
        )

        self.usage_tracker = ToolUsageTracker()
        self.session_manager = SessionManager()
        self._session_routes: Dict[str, IsaacSessionRoute] = {}
        # The one session that unauthenticated requests on the sessionless MCP
        # protocol (2026-07-28) share over stdio, where this server serves a
        # single client; see _get_request_session_id.
        self._sessionless_session_id = str(uuid.uuid4())
        # Handshake-era sessions whose end is hooked to drop their state; see
        # _watch_session_end.
        self._watched_sessions: Set[str] = set()
        # Backends that keep their own per-session state (the headless USD
        # stages) drop it here; see add_session_end_hook.
        self._session_end_hooks: List[Callable[[str], None]] = []
        self._read_only_tools: Dict[str, bool] = {}
        # Arbitrary-code tools security.allow_script_execution kept off the
        # surface; a call to one answers ScriptExecutionDisabled.
        self._hidden_script_tools: Set[str] = set()
        self._isaac_instance_locks: Dict[str, asyncio.Lock] = {}
        # Token buckets keyed by (agent, tool); the per-agent ceiling across
        # every tool uses _GLOBAL_RATE_BUCKET as the tool key.
        self._rate_limiters: Dict[Tuple[str, str], RateLimiter] = {}
        # Tools that skip the per-tool bucket and spend only the agent's
        # ceiling; see exempt_from_per_tool_rate_limit.
        self._per_tool_rate_exempt: Set[str] = set()
        # Paged tools whose follow-on pages (offset > 0) skip the per-tool
        # bucket; see exempt_paging_continuations.
        self._paging_continuation_exempt: Dict[str, str] = {}
        self._rate_limit_enabled = self.settings.security.rate_limiting_enabled
        self._rate_limit_rate = self.settings.security.requests_per_minute / 60.0
        self._rate_limit_burst = self.settings.security.burst_size
        global_per_minute = self.settings.security.global_requests_per_minute
        self._global_rate_limit_rate = global_per_minute / 60.0
        # Ten seconds of the per-agent allowance, so an agent fanning out
        # across many tools at once is not refused by a ceiling meant for
        # runaway loops.
        self._global_rate_limit_burst = max(
            self._rate_limit_burst, global_per_minute // 6
        )
        # How long a call waits for an instance already in use. Generous enough
        # for ordinary contention, short enough that a caller is not left
        # guessing through a 1000-frame step.
        self._instance_lock_timeout = float(self.settings.server.timeout)

        # One adapter per registered backend; None when its runtime is not
        # importable in this process or the backend was not selected. The
        # factories import their backend's modules when called, so a disabled
        # backend's runtime (bpy, pxr, aiohttp) is never probed or imported.
        # Isaac's adapter is always built: it owns the default client the
        # instance registry and IsaacTools hold, and building it dials nothing.
        self._adapters: Dict[str, Optional[BackendAdapter]] = {
            spec.name: (
                spec.adapter_factory(self.settings)
                if spec.name == "isaac" or self._backend_enabled(spec.name)
                else None
            )
            for spec in BACKENDS
        }
        # Which backend registered each tool, for the CLI's grouping.
        self._tool_backends: Dict[str, str] = {}

        # Multi-instance Isaac Sim registry. The adapter owns the default
        # instance's client and builds the others from the same settings.
        self._isaac_clients: Dict[str, IsaacSocketClient] = {}
        self._active_instance: str = "default"
        default_client = self.isaac_adapter.client
        self._isaac_clients["default"] = default_client
        for inst in self.settings.isaac_sim.instances:
            self._isaac_clients[inst.name] = self._build_isaac_client(
                socket_host=inst.host,
                socket_port=inst.port,
                socket_timeout=inst.timeout,
                bridge_enabled=inst.bridge_enabled,
                bridge_host=inst.bridge_host,
                bridge_port=inst.bridge_port,
                bridge_timeout=inst.bridge_timeout,
                bridge_fallback_to_vscode=inst.bridge_fallback_to_vscode,
                socket_protocol=inst.socket_protocol,
                socket_auth_token=inst.socket_auth_token,
            )
        self.client = default_client
        self._isaac_tools = IsaacTools(
            self.client,
            self.settings,
            client_resolver=self._get_request_isaac_client,
        )

        # Decided before registration: the backends whose tools register.
        self._registered_backends: Tuple[str, ...] = tuple(
            spec.name for spec in BACKENDS if self._registers(spec.name)
        )

        self.mcp = FastMCP(name="simul", version=_PACKAGE_VERSION)
        # Decided before registration, since _task_optional reads it.
        self._tasks_enabled = self._register_tasks_extension()

        # Tag every CallTool request with a fresh correlation id and emit an
        # audit row. Done as FastMCP middleware (vs. wrapping each tool) so it
        # composes with all decorator overloads, sync + async tool bodies, and
        # any future tools without per-registration changes.
        from ..logging import build_request_context_middleware

        self.mcp.add_middleware(build_request_context_middleware())
        # Inside the audit middleware, so the audit row sees the envelope.
        self.mcp.add_middleware(build_error_envelope_middleware(self))

        self._register_tools()

        # Written once the tools exist, so the instructions describe exactly
        # what tools/list shows: the registered backends, and the SimReady
        # helpers only when one registered (the thin surfaces have none).
        if _FASTMCP_SUPPORTS_INSTRUCTIONS:
            self.mcp.instructions = build_instructions(
                self._registered_backends, self._tool_backends
            )

        self.logger.info("Simul 3D MCP Server initialized")

    @property
    def isaac_adapter(self) -> IsaacRuntimeAdapter:
        """The Isaac Sim adapter; always built, since its client has no import-time dependency."""
        adapter = self._adapters["isaac"]
        if not isinstance(adapter, IsaacRuntimeAdapter):
            raise RuntimeError("The Isaac backend adapter is missing from the registry")
        return adapter

    @property
    def headless_adapter(self) -> Optional[BackendAdapter]:
        """The headless USD adapter, or None when pxr is not importable."""
        return self._adapters["usd"]

    @headless_adapter.setter
    def headless_adapter(self, adapter: Optional[BackendAdapter]) -> None:
        self._adapters["usd"] = adapter

    @property
    def blender_adapter(self) -> Optional[BackendAdapter]:
        """The attached/embedded Blender adapter, or None when unavailable."""
        return self._adapters["blender"]

    @blender_adapter.setter
    def blender_adapter(self, adapter: Optional[BackendAdapter]) -> None:
        self._adapters["blender"] = adapter

    @property
    def unreal_adapter(self) -> Optional[BackendAdapter]:
        """The Unreal adapter, or None when its module did not import."""
        return self._adapters["unreal"]

    @unreal_adapter.setter
    def unreal_adapter(self, adapter: Optional[BackendAdapter]) -> None:
        self._adapters["unreal"] = adapter

    def tools_by_backend(self) -> Dict[str, List[str]]:
        """Group the registered tool names by the backend that registered them.

        Returns:
            Backend label to sorted tool names, in registry order, followed by
            a ``Server`` group for the tools no backend owns (usage stats).
        """
        groups: Dict[str, List[str]] = {}
        for spec in BACKENDS:
            names = sorted(
                name for name, backend in self._tool_backends.items() if backend == spec.name
            )
            if names:
                groups[spec.label] = names
        unowned = sorted(self._registered_tool_names() - set(self._tool_backends))
        if unowned:
            groups["Server"] = unowned
        return groups

    def _registered_tool_names(self) -> Set[str]:
        """Return every tool name currently registered on the FastMCP instance.

        Returns:
            The names, read from FastMCP's local provider where components
            are keyed ``tool:<name>@``; empty when the provider is absent.
        """
        provider = getattr(self.mcp, "local_provider", None)
        components = getattr(provider, "_components", {}) if provider is not None else {}
        return {
            key.split(":", 1)[1].split("@", 1)[0] for key in components if key.startswith("tool:")
        }

    def _get_rate_limiter(self, tool_name: str, agent_id: str) -> RateLimiter:
        """Return the token bucket for one agent's use of one tool.

        Args:
            tool_name: Registered tool name, or ``_GLOBAL_RATE_BUCKET`` for
                the agent's ceiling across every tool.
            agent_id: Stable id of the calling agent.

        Returns:
            The bucket, created on first use.
        """
        key = (agent_id, tool_name)
        limiter = self._rate_limiters.get(key)
        if limiter is None:
            if len(self._rate_limiters) >= _RATE_LIMITER_SWEEP_SIZE:
                self._sweep_idle_rate_limiters()
            if tool_name == _GLOBAL_RATE_BUCKET:
                limiter = RateLimiter(
                    self._global_rate_limit_rate, self._global_rate_limit_burst
                )
            else:
                limiter = RateLimiter(self._rate_limit_rate, self._rate_limit_burst)
            self._rate_limiters[key] = limiter
            session_id = self._get_request_session_id()
            if session_id is not None:
                self._watch_session_end(session_id)
        return limiter

    def exempt_from_per_tool_rate_limit(self, tool_names: Iterable[str]) -> None:
        """Count ``tool_names`` only against each agent's ceiling across all tools.

        For cheap local queries an agent issues once per item while it works
        through a scene (the headless USD per-prim queries): a per-tool
        bucket of ``security.requests_per_minute`` with ``burst_size`` refused
        most of a prim-by-prim inspection, while the per-agent ceiling
        (``security.global_requests_per_minute``) still stops a runaway loop.

        Args:
            tool_names: Registered tool names.
        """
        self._per_tool_rate_exempt.update(tool_names)

    def exempt_paging_continuations(self, tool_names: Iterable[str]) -> None:
        """Count the follow-on pages of paged ``tool_names`` only against each agent's ceiling.

        Walking a listing page by page is one request spread over many calls:
        the first page spends a token from the tool's own bucket, and each
        call carrying a nonzero offset (``next_offset`` of the page before,
        the parameter ``paged_results.PAGED_RESULTS`` declares) counts only
        against the per-agent ceiling (``security.global_requests_per_minute``),
        which still stops a runaway loop. With the per-tool bucket alone a
        25-page walk of a large stage was refused 15 times.

        Args:
            tool_names: Registered names of paged tools.
        """
        from .paged_results import paging_for

        for name in tool_names:
            paging = paging_for(name)
            if paging is None:
                raise ValueError(f"{name} declares no page in paged_results.PAGED_RESULTS")
            self._paging_continuation_exempt[name] = paging.offset_param

    def _is_paging_continuation(self, tool_name: str, params: Optional[Mapping[str, Any]]) -> bool:
        offset_param = self._paging_continuation_exempt.get(tool_name)
        if offset_param is None or not params:
            return False
        offset = params.get(offset_param)
        return isinstance(offset, int) and not isinstance(offset, bool) and offset > 0

    def _sweep_idle_rate_limiters(self) -> None:
        """Drop the buckets that have refilled to their burst size.

        Such a bucket would be rebuilt identical on the agent's next call, so
        the sweep never changes a verdict; it keeps the table from growing
        with every identity that has ever called.
        """
        now = time.monotonic()
        idle = [
            key
            for key, limiter in self._rate_limiters.items()
            if limiter.tokens + (now - limiter.last_update) * limiter.rate >= limiter.burst
        ]
        for key in idle:
            del self._rate_limiters[key]

    def _request_connection(self) -> Any:
        """Return the handshake-era MCP connection the current request arrived on.

        Only such a connection is one client session whose end can be
        observed: on the sessionless protocol every request is its own
        connection, closed when the request returns.

        Returns:
            The SDK ``Connection`` (it carries the ``exit_stack`` unwound when
            the session closes), or None.
        """
        ctx = self._get_request_context()
        request = getattr(ctx, "request_context", None) if ctx is not None else None
        if request is None:
            return None
        if getattr(request, "protocol_version", None) in MODERN_PROTOCOL_VERSIONS:
            return None
        # FastMCP reads the same private attribute for its per-session state.
        connection = getattr(getattr(request, "session", None), "_connection", None)
        if getattr(connection, "exit_stack", None) is None:
            return None
        return connection

    def _watch_session_end(self, session_id: str) -> None:
        """Arrange for the state kept for ``session_id`` to be dropped when its session closes.

        Called whenever per-session state (a rate-limit bucket, an Isaac
        route) is created, so each handshake-era session is hooked once. A
        request whose session end cannot be observed (stdio's sessionless
        session, an authenticated sessionless user) keeps its state for the
        life of the server, bounded by the number of such identities.
        """
        if session_id in self._watched_sessions:
            return
        connection = self._request_connection()
        if connection is None:
            return
        self._watched_sessions.add(session_id)
        connection.exit_stack.callback(self._forget_session, session_id)

    def _forget_session(self, session_id: str) -> None:
        """Drop what the server keeps for an ended MCP session.

        Its Isaac route and bindings and its rate-limit buckets go. Its claims
        stay in the session records until they expire, so another agent cannot
        take the instance from one that is reconnecting, but they are marked
        orphaned, which lets the same ``agent_id`` reclaim them at once.
        """
        self._watched_sessions.discard(session_id)
        route = self._session_routes.pop(session_id, None)
        if route is not None:
            owner = self._owner_for_session(session_id)
            for port in {binding.port for binding in route.bindings.values()}:
                try:
                    self.session_manager.get_instance_session(port).mark_orphaned(owner)
                except Exception as exc:
                    self.logger.warning(
                        "Could not mark the ended session's claims on port %s orphaned: %s",
                        port,
                        exc,
                    )
        label = default_agent_label(session_id)
        for key in [key for key in self._rate_limiters if key[0] == label]:
            del self._rate_limiters[key]
        self.usage_tracker.forget_session(label)
        for hook in self._session_end_hooks:
            try:
                hook(session_id)
            except Exception as exc:
                self.logger.warning(
                    "Session-end cleanup %s failed for session %s: %s",
                    getattr(hook, "__qualname__", hook),
                    session_id,
                    exc,
                )

    def add_session_end_hook(self, hook: Callable[[str], None]) -> None:
        """Have ``hook(session_id)`` run when a handshake-era MCP session ends.

        For a backend that keeps state of its own per session: it registers
        the hook once, and calls ``_watch_session_end`` for the request's
        session whenever it creates such state, so the end is observed. A
        session whose end cannot be observed (see ``_watch_session_end``)
        keeps its state for the life of the server.

        Args:
            hook: Called with the ended session's id; an exception is logged.
        """
        self._session_end_hooks.append(hook)

    def _check_rate_limit(
        self, tool_name: str, agent_id: Optional[str] = None, continuation: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Spend one token for ``tool_name`` on behalf of the calling agent.

        Two buckets must both have room: the agent's bucket for this tool and
        the agent's ceiling across every tool. Keying on the agent means one
        looping agent cannot starve another on the same tool, and the ceiling
        means it cannot dodge a per-tool refusal by switching to
        execute_isaac_script. A refusal spends nothing, so a caller that waits
        ``retry_after_seconds`` finds the token it was promised.

        Args:
            tool_name: Registered tool name.
            agent_id: Calling agent; resolved from the MCP session when omitted.
            continuation: The call fetches a follow-on page of a tool
                registered with ``exempt_paging_continuations``; only the
                agent's ceiling applies.

        Returns:
            The RateLimitError payload carrying ``retry_after_seconds``, or
            None when the call may proceed.
        """
        if not self._rate_limit_enabled or tool_name in _RATE_LIMIT_EXEMPT:
            return None
        agent = agent_id or self._resolve_agent_id(None)
        tool_bucket = (
            None
            if continuation or tool_name in self._per_tool_rate_exempt
            else self._get_rate_limiter(tool_name, agent)
        )
        # An anonymous HTTP client is named by its connection, so its per-tool
        # buckets start afresh when it reconnects; the ceiling across every
        # tool is kept per host, so reconnecting for each request is
        # throttled all the same.
        agent_bucket = self._get_rate_limiter(_GLOBAL_RATE_BUCKET, _ceiling_key(agent))
        tool_wait = tool_bucket.seconds_until_available() if tool_bucket is not None else 0.0
        agent_wait = agent_bucket.seconds_until_available()
        if tool_wait <= 0.0 and agent_wait <= 0.0:
            if tool_bucket is not None:
                tool_bucket.acquire()
            agent_bucket.acquire()
            return None

        per_tool = tool_wait >= agent_wait
        retry_after = max(round(max(tool_wait, agent_wait), 3), 0.001)
        security = self.settings.security
        payload = ErrorResponse(
            error=f"Rate limit exceeded; retry in {retry_after} s",
            error_type=error_types.RATE_LIMIT_ERROR,
            details={
                "tool": tool_name,
                "agent_id": agent,
                "scope": "tool" if per_tool else "agent",
                "limit_per_minute": (
                    security.requests_per_minute
                    if per_tool
                    else security.global_requests_per_minute
                ),
            },
        ).model_dump()
        payload["retry_after_seconds"] = retry_after
        return payload

    def _get_request_context(self) -> Any:
        """Return the active FastMCP request context when available."""
        if _current_context is None:
            return None
        try:
            return _current_context.get()
        except Exception:
            return None

    def _get_request_session_id(self) -> Optional[str]:
        """Return the id of the MCP session the current request belongs to.

        A connection opened with the ``initialize`` handshake (protocol
        2025-11-25 and earlier) is one session, and FastMCP reports its id.

        The 2026-07-28 protocol has no sessions. FastMCP gives each of its
        requests a fresh id, taken over HTTP from whatever ``Mcp-Session-Id``
        header the client sends, so that id is never used: it would make every
        call a new agent, and a client could name another's session. Such a
        request is scoped the way FastMCP scopes its own per-client state: to
        the authenticated user when it carries one; on stdio, where the
        server serves a single client, to one session per server; and over
        unauthenticated HTTP or SSE, where nothing tells one client from
        another, to no session at all (see ``_request_is_anonymous``).

        Returns:
            The session id, or None outside an MCP request and for a request
            without a client identity.
        """
        ctx = self._get_request_context()
        if ctx is None:
            return None
        try:
            request = getattr(ctx, "request_context", None)
            if getattr(request, "protocol_version", None) not in MODERN_PROTOCOL_VERSIONS:
                return ctx.session_id
            principal = _request_principal()
            if principal is not None:
                digest = hashlib.sha256(principal.encode("utf-8")).hexdigest()
                return f"user-{digest[:32]}"
            if ctx.transport in _CONTEXT_NETWORK_TRANSPORTS:
                return None
            return self._sessionless_session_id
        except Exception:
            return None

    def _request_is_anonymous(self) -> bool:
        """Report whether the current MCP request carries no client identity.

        That is an unauthenticated request on the sessionless protocol over
        HTTP or SSE (or one whose identity could not be read). The server
        cannot tell its client from any other, so it keeps nothing per client
        for it: the request targets the server's default Isaac instance,
        cannot switch instance or claim one, finds every live claim held by
        someone else, and spends the shared anonymous rate-limit buckets.
        """
        return self._get_request_context() is not None and self._get_request_session_id() is None

    def _session_required_error(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Refuse a tool that keeps state per client when the request has no client identity.

        Args:
            tool_name: The tool being called, for the message.

        Returns:
            The SessionRequired envelope when ``_request_is_anonymous``,
            otherwise None.
        """
        if not self._request_is_anonymous():
            return None
        return ErrorResponse(
            error=(
                f"{tool_name} keeps state for the calling client, and this request "
                "carries no client identity: an unauthenticated HTTP request on the "
                "sessionless MCP protocol (2026-07-28) has no session, so the server "
                "cannot tell its client from any other."
            ),
            error_type=error_types.SESSION_REQUIRED,
            details={
                "active_instance": self._active_instance,
                "hint": (
                    "Connect with the initialize handshake (MCP protocol 2025-11-25 or "
                    'earlier, e.g. fastmcp.Client(url, mode="legacy")) or over stdio. '
                    "Until then the Isaac tools target the server's default instance, "
                    "named in active_instance."
                ),
            },
        ).model_dump()

    def _get_or_create_session_route(self, session_id: str) -> IsaacSessionRoute:
        """Return the per-session Isaac routing state."""
        route = self._session_routes.get(session_id)
        if route is None:
            route = IsaacSessionRoute(
                session_id=session_id,
                active_instance=self._active_instance,
            )
            self._session_routes[session_id] = route
            self._watch_session_end(session_id)
        return route

    def _get_request_route(self) -> Optional[IsaacSessionRoute]:
        """Return the Isaac routing state for the active MCP session."""
        session_id = self._get_request_session_id()
        if session_id is None:
            return None
        return self._get_or_create_session_route(session_id)

    def _get_effective_instance_name(self) -> str:
        """Resolve the Isaac instance targeted by the current request."""
        route = self._get_request_route()
        if route is not None:
            return route.active_instance
        return self._active_instance

    def _get_request_isaac_client(self) -> IsaacSocketClient:
        """Resolve the Isaac socket client for the current request."""
        instance_name = self._get_effective_instance_name()
        return self._isaac_clients.get(instance_name, self.client)

    def _get_instance_lock(self, instance_name: str) -> asyncio.Lock:
        """Return the per-instance execution lock."""
        if instance_name not in self._isaac_instance_locks:
            self._isaac_instance_locks[instance_name] = asyncio.Lock()
        return self._isaac_instance_locks[instance_name]

    def _set_request_active_instance(self, instance_name: str) -> None:
        """Set the active Isaac instance for the current MCP session."""
        route = self._get_request_route()
        if route is not None:
            route.active_instance = instance_name
            return
        self._switch_active_instance(instance_name)

    def _resolve_agent_id(self, agent_id: Optional[str]) -> str:
        """Resolve a stable per-session agent id (see ``default_agent_label``)."""
        if agent_id:
            return agent_id
        session_id = self._get_request_session_id()
        if session_id:
            return default_agent_label(session_id)
        if self._get_request_context() is not None:
            # A request without a client identity; see _request_is_anonymous.
            return self._anonymous_agent()
        return f"agent-{id(self):x}"

    def _anonymous_agent(self) -> str:
        """Name the client of a request without a client identity, for budgets and records only.

        Over HTTP that is the TCP peer, ``anonymous@<host>:<port>``: assigned
        by the network stack, not chosen by the client, and stable for as long
        as the client keeps its connection open, so each client gets its own
        rate-limit budget (three sessionless clients used to share one and be
        refused half their calls). It is no security boundary: a client that
        reconnects gets a new budget, as one opening a new session does, and
        nothing per-client (an instance switch, a claim) is kept for it.

        Returns:
            The peer's name, or ``anonymous`` when there is no HTTP request.
        """
        try:
            from fastmcp.server.dependencies import get_http_request

            peer = get_http_request().client
        except Exception:
            return _ANONYMOUS_AGENT
        if peer is None:
            return _ANONYMOUS_AGENT
        return f"{_ANONYMOUS_AGENT}@{peer.host}:{peer.port}"

    def _client_label(self) -> Optional[str]:
        """Return the accounting label an anonymous HTTP client sends in ``X-Simul-Agent``.

        Recorded as the call's agent (``anonymous:<label>``) in the usage and
        audit logs so an operator can tell clients apart; never used to decide
        anything, since any client can send any label.
        """
        try:
            from fastmcp.server.dependencies import get_http_request

            raw = get_http_request().headers.get(_CLIENT_LABEL_HEADER)
        except Exception:
            return None
        if not raw:
            return None
        label = "".join(c for c in str(raw) if c.isprintable() and not c.isspace())[:64]
        return label or None

    def _bind_request_session(
        self,
        instance_name: str,
        port: int,
        agent_id: str,
        purpose: Optional[str] = None,
    ) -> IsaacSessionBinding:
        """Create or update the current session's binding for an instance."""
        session_id = self._get_request_session_id()
        binding_id = f"{agent_id}:{instance_name}"
        binding = IsaacSessionBinding(
            binding_id=binding_id,
            session_id=session_id or agent_id,
            instance_name=instance_name,
            agent_id=agent_id,
            port=port,
            purpose=purpose,
        )
        route = self._get_request_route()
        if route is not None:
            route.bindings[instance_name] = binding
            route.active_instance = instance_name
        return binding

    def _get_active_binding(self) -> Optional[IsaacSessionBinding]:
        """Return the binding for the request's active Isaac instance."""
        route = self._get_request_route()
        if route is None:
            return None
        return route.bindings.get(route.active_instance)

    def _release_request_binding(
        self,
        instance_name: str,
        agent_id: Optional[str] = None,
    ) -> Optional[IsaacSessionBinding]:
        """Remove and return the current session's binding for an instance."""
        route = self._get_request_route()
        if route is None:
            return None
        binding = route.bindings.get(instance_name)
        if binding is None:
            return None
        if agent_id is not None and binding.agent_id != agent_id:
            return None
        route.bindings.pop(instance_name, None)
        if route.active_instance == instance_name and route.bindings:
            route.active_instance = next(iter(route.bindings))
        return binding

    def _request_agent_id(self) -> str:
        """Return the agent id the current request acts as: its binding's, else the session's."""
        binding = self._get_active_binding()
        if binding is not None:
            return binding.agent_id
        return self._resolve_agent_id(None)

    def _recorded_agent_id(self, session_agent_id: str) -> str:
        """Return the agent a usage record names: the active claim's label, else the session's.

        Reads the session's route without creating one, so a call that never
        touched Isaac adds no routing state.
        """
        session_id = self._get_request_session_id()
        if session_id is None and self._get_request_context() is not None:
            label = self._client_label()
            return f"{_ANONYMOUS_AGENT}:{label}" if label else session_agent_id
        route = self._session_routes.get(session_id) if session_id is not None else None
        binding = route.bindings.get(route.active_instance) if route is not None else None
        return binding.agent_id if binding is not None else session_agent_id

    def _claim_owner(self) -> str:
        """Return the opaque owner token for claims made by the current request.

        A claim is bound to the MCP session that made it, because ``agent_id``
        is chosen by the caller and shown to every agent by
        ``list_isaac_instances``. The token is a digest of the transport's
        session id, so publishing it in session records reveals nothing, or of
        this server process when the request carries no session.
        """
        return self._owner_for_session(self._get_request_session_id())

    def _owner_for_session(self, session_id: Optional[str]) -> str:
        """Return the opaque claim owner token of ``session_id`` (see ``_claim_owner``)."""
        source = f"session:{session_id}" if session_id else f"process:{os.getpid()}:{id(self):x}"
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]

    def _find_foreign(
        self,
        sessions: List[Dict[str, Any]],
        caller_agent_id: str,
        *,
        allow_takeover: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Return the first record in ``sessions`` that is not the caller's own claim.

        A claim is foreign when its ``agent_id`` differs from the caller's or
        when a different MCP session made it (its ``owner`` token differs), so
        passing another agent's ``agent_id`` does not inherit that agent's
        claim. A request without a client identity holds no claim, so every
        record is foreign to it.

        Args:
            sessions: Live session records of one instance.
            caller_agent_id: The agent making the request.
            allow_takeover: The caller is claiming: a record of the same
                ``agent_id`` that is orphaned (its session ended or its server
                process exited, so nothing else can renew it) is not foreign.
                This is how an agent reclaims its own instance after a
                reconnect or a server restart instead of waiting out the TTL.

        Returns:
            The blocking record, or None.
        """
        if self._request_is_anonymous():
            return sessions[0] if sessions else None
        owner = self._claim_owner()
        raw_session = self._get_request_session_id()
        for session in sessions:
            holder_owner = session.get("owner")
            if holder_owner is None and raw_session and session.get("agent_id") == raw_session:
                # Written by an older server, whose default label was the raw
                # session id: still this session's own claim until it expires.
                continue
            if session.get("agent_id") != caller_agent_id:
                return session
            if holder_owner is not None and holder_owner != owner:
                if allow_takeover and claim_is_orphaned(session):
                    continue
                return session
        return None

    def _holds_claim(self, view: ClaimView, instance_name: str, caller_agent_id: str) -> bool:
        """Report whether the caller can show a claim of its own on the instance.

        Used while a damaged record file is being repaired: a caller passes
        when the store already holds its record, or when its session's binding
        to the instance (kept in this process) shows it held one.
        """
        if self._request_is_anonymous():
            return False
        owner = self._claim_owner()
        if any(
            s.get("agent_id") == caller_agent_id and s.get("owner") == owner
            for s in view.sessions
        ):
            return True
        route = self._get_request_route()
        binding = route.bindings.get(instance_name) if route is not None else None
        return binding is not None and binding.agent_id == caller_agent_id

    @staticmethod
    def _unverified_holder(view: ClaimView) -> Dict[str, Any]:
        """Stand in for the claim a repaired record file may have lost."""
        return {
            "agent_id": None,
            "purpose": "",
            "last_active": view.repaired_at,
            "unverified": True,
        }

    def _claim_conflict(self, instance_name: str, caller_agent_id: str) -> ClaimConflict:
        """Build the check ``InstanceSession.register`` runs, under the store lock, for a claim.

        The caller may claim when no other agent holds a live claim
        (an orphaned claim of its own ``agent_id`` it may take over) and, while
        the record file is being repaired, only when it can show a claim of
        its own (``_holds_claim``).
        """

        def conflict(view: ClaimView) -> Optional[Dict[str, Any]]:
            holder = self._find_foreign(view.sessions, caller_agent_id, allow_takeover=True)
            if holder is not None:
                return holder
            if view.repaired_at is not None and not self._holds_claim(
                view, instance_name, caller_agent_id
            ):
                return self._unverified_holder(view)
            return None

        return conflict

    def _foreign_claim(
        self, instance_name: str, caller_agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the live claim another agent holds on ``instance_name``, when enforced.

        Enforcement fails closed. Records that cannot be read stand for a
        claim nobody can verify, and so does the TTL after a damaged record
        file was repaired, for every caller that cannot show a claim of its
        own; a caller whose binding shows it held one re-asserts its record
        here, so the holders of the lost claims keep working through the
        repair.

        Args:
            instance_name: Registry name of the Isaac instance being used.
            caller_agent_id: The agent making the request.

        Returns:
            The holder's session record when ``isaac_sim.enforce_claims`` is on
            and a live claim by a different agent exists (see
            ``_find_foreign``); a record with ``unverified`` set, and an
            ``error`` when the store could not be read, when the claims cannot
            be verified; otherwise ``None``. Expired claims are pruned by the
            session manager, so a stale holder never blocks.
        """
        if not self.settings.isaac_sim.enforce_claims:
            return None
        client = self._isaac_clients.get(instance_name)
        if client is None:
            return None
        store = self.session_manager.get_instance_session(client._port)
        try:
            view = store.view()
        except ClaimStoreError as exc:
            return {"agent_id": None, "purpose": "", "unverified": True, "error": str(exc)}
        holder = self._find_foreign(view.sessions, caller_agent_id)
        if holder is not None or view.repaired_at is None:
            return holder
        if not self._holds_claim(view, instance_name, caller_agent_id):
            return self._unverified_holder(view)
        route = self._get_request_route()
        binding = route.bindings.get(instance_name) if route is not None else None
        if binding is not None and not any(
            s.get("agent_id") == caller_agent_id for s in view.sessions
        ):
            try:
                result = store.register(
                    caller_agent_id,
                    binding.purpose or "",
                    owner=self._claim_owner(),
                    conflict=self._claim_conflict(instance_name, caller_agent_id),
                )
            except ClaimStoreError as exc:
                return {"agent_id": None, "purpose": "", "unverified": True, "error": str(exc)}
            if result["status"] == "refused":
                return cast(Dict[str, Any], result["holder"])
            self.logger.warning(
                "Re-asserted the claim of agent %r on port %s after its records were repaired",
                caller_agent_id,
                client._port,
            )
        return None

    def _claimed_error(
        self,
        instance_name: str,
        holder: Dict[str, Any],
        caller_agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the InstanceClaimed payload naming the holder and the way forward.

        A holder with ``unverified`` set stands for claims that cannot be
        checked: unreadable records (``ClaimStoreError``) or a repaired record
        file (InstanceClaimed with no holder until the TTL passes).
        """
        port = self._port_of(instance_name)
        clear_hint = (
            "An operator can list and clear claims with `simul isaac claims` "
            f"(`simul isaac claims --clear --port {port}`)."
        )
        if holder.get("unverified"):
            return self._unverified_claims_error(instance_name, holder, clear_hint)
        expires_in = max(0.0, CLAIM_TTL_SECONDS - (time.time() - float(holder.get("last_active", 0.0))))
        same_label = caller_agent_id is not None and holder.get("agent_id") == caller_agent_id
        impersonation = (
            f" The agent_id {caller_agent_id!r} is bound to a different MCP session;"
            " pick a unique agent_id or omit it."
            if same_label
            else ""
        )
        orphaned = claim_is_orphaned(holder)
        hint = (
            "Read-only tools still work. Ask the holder to call "
            "release_isaac_instance, or wait for the claim to expire "
            f"({int(CLAIM_TTL_SECONDS)} s without activity), then call "
            "claim_isaac_instance yourself. list_isaac_instances shows other instances."
        )
        if orphaned:
            hint += (
                " The holder's MCP session has ended: calling claim_isaac_instance with "
                f"agent_id {holder.get('agent_id')!r} takes the claim over now."
            )
        details: Dict[str, Any] = {
            "instance": instance_name,
            "holder_agent_id": holder.get("agent_id"),
            "holder_purpose": holder.get("purpose", ""),
            "claim_expires_in_seconds": round(expires_in, 1),
            "hint": f"{hint} {clear_hint}",
        }
        if orphaned:
            details["holder_orphaned"] = True
        return ErrorResponse(
            error=(
                f"Isaac instance {instance_name!r} is claimed by agent "
                f"{holder.get('agent_id')!r} for {holder.get('purpose', '')!r}; "
                "isaac_sim.enforce_claims refuses mutating tools from other agents while "
                "that claim is live." + impersonation
            ),
            error_type=error_types.INSTANCE_CLAIMED,
            details=details,
        ).model_dump()

    def _unverified_claims_error(
        self, instance_name: str, holder: Dict[str, Any], clear_hint: str
    ) -> Dict[str, Any]:
        """Build the refusal for claims enforcement cannot verify (see ``_claimed_error``)."""
        if holder.get("error"):
            return ErrorResponse(
                error=(
                    f"The claims on Isaac instance {instance_name!r} cannot be read, so "
                    "isaac_sim.enforce_claims refuses mutating tools: "
                    f"{holder['error']}"
                ),
                error_type=ClaimStoreError.error_type,
                details={
                    "instance": instance_name,
                    "hint": (
                        "Read-only tools still work. Check that the session directory "
                        f"({self.session_manager.session_dir}) is writable by this user."
                    ),
                },
            ).model_dump()
        repaired_at = holder.get("last_active")
        expires_in = (
            max(0.0, CLAIM_TTL_SECONDS - (time.time() - float(repaired_at)))
            if isinstance(repaired_at, (int, float))
            else CLAIM_TTL_SECONDS
        )
        return ErrorResponse(
            error=(
                f"The claim records of Isaac instance {instance_name!r} were damaged and "
                "repaired, and a claim they held may still be live, so "
                "isaac_sim.enforce_claims refuses mutating tools from agents without a "
                "claim of their own until the claim TTL has passed."
            ),
            error_type=error_types.INSTANCE_CLAIMED,
            details={
                "instance": instance_name,
                "holder_agent_id": None,
                "holder_purpose": "",
                "claim_expires_in_seconds": round(expires_in, 1),
                "hint": f"Read-only tools still work. Retry after the claim TTL. {clear_hint}",
            },
        ).model_dump()

    def _port_of(self, instance_name: str) -> Optional[int]:
        """Return the socket port the session records of ``instance_name`` are keyed on."""
        client = self._isaac_clients.get(instance_name)
        return client._port if client is not None else None

    def _claim_store_error(self, exc: ClaimStoreError, instance_name: str) -> Dict[str, Any]:
        """Build the error envelope for a claim tool whose records could not be saved."""
        self.logger.error("Claim records for %r could not be saved: %s", instance_name, exc)
        return ErrorResponse(
            error=f"The claim was not saved: {exc}",
            error_type=ClaimStoreError.error_type,
            details={
                "instance": instance_name,
                "path": str(exc.path),
                "hint": (
                    "Check that the session directory "
                    f"({self.session_manager.session_dir}) is writable by this user."
                ),
            },
        ).model_dump()

    async def _tool_parameters(self, tool_name: str) -> List[str]:
        """Return the argument names a registered tool's input schema lists (empty if unknown)."""
        get_tool = getattr(self.mcp, "get_tool", None)
        if not callable(get_tool):
            return []
        try:
            tool = get_tool(tool_name)
            if inspect.isawaitable(tool):
                tool = await tool
        except Exception:
            return []
        schema = getattr(tool, "parameters", None)
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        return [str(name) for name in properties]

    async def _tool_is_read_only(self, tool_name: str) -> bool:
        """Report whether the registered tool carries ``readOnlyHint``; unknown tools count as mutating."""
        cached = self._read_only_tools.get(tool_name)
        if cached is not None:
            return cached
        get_tool = getattr(self.mcp, "get_tool", None)
        read_only = False
        if callable(get_tool):
            tool = get_tool(tool_name)
            if inspect.isawaitable(tool):
                tool = await tool
            annotations = getattr(tool, "annotations", None)
            if isinstance(annotations, dict):
                read_only = bool(annotations.get("readOnlyHint"))
            else:
                # MCP SDK v2 field name; the wire keeps the camelCase alias.
                read_only = bool(getattr(annotations, "read_only_hint", False))
        self._read_only_tools[tool_name] = read_only
        return read_only

    async def _run_envelope(
        self,
        tool_name: str,
        call: Callable[[Optional[BaseModel]], Awaitable[Dict[str, Any]]],
        *,
        admission: Admission = _admit_every_call,
        request_model: Optional[Type[BaseModel]] = None,
        request_kwargs: Optional[Mapping[str, Any]] = None,
        response_model: Optional[Type[BaseModel]] = None,
        params: Optional[Dict[str, Any]] = None,
        script_sha256: Optional[str] = None,
        discard: Optional[Callable[[], None]] = None,
        input_echo: Optional[InputEcho] = None,
    ) -> ToolResult:
        """Run one tool call through the envelope every backend shares.

        The stages, in order: the per-agent rate limit; the backend's
        ``admission`` hook (Isaac's claim enforcement and instance lock);
        validation of ``request_kwargs`` against ``request_model``; the call;
        success normalisation (``apply_success_from_error``) and, for a
        successful payload, the ``response_model`` schema; one usage record;
        then the result budget and single transmission. A refusal ends the
        call at its stage, and an exception from the call becomes an
        ErrorResponse named after the exception, so every call that returns
        a result, refusals and invalid input included, is rate limited,
        recorded exactly once and sent once.

        Two exits return no result: cancellation (the client gave up while the
        call ran or while it waited for Isaac's instance lock), which is
        recorded as ``cancelled`` and re-raised, and an exception raised by
        the admission hook itself, which
        propagates to FastMCP unrecorded. A cancelled call's record says
        whether the engine ran the operation all the same (``executed``):
        false when it never started, true with the engine's own outcome when
        the call waited for the engine to finish (Isaac), ``"unknown"`` when
        the request was already sent and nothing waited for it (Unreal,
        Blender), since cancelling the request does not stop the engine.
        ``discard`` runs on every exit before ``call`` starts, a cancelled
        wait and an admission exception included.
        Every record carries the call's request id, and the call's audit row
        is told its status, ``error_type`` and agent (``note_call_outcome``).

        Args:
            tool_name: Registered tool name, for the rate limit, the logs and
                the usage record.
            call: Performs the operation. Receives the validated request, or
                None without a ``request_model``, and returns the payload or
                an error envelope.
            admission: The backend's gate between the rate limit and the call.
            request_model: Model the tool's arguments are validated against
                before ``call`` runs.
            request_kwargs: The arguments to validate.
            response_model: Schema a successful payload is validated against;
                an error envelope keeps its own fields.
            params: Call parameters worth keeping in the usage log.
            script_sha256: Digest of the agent-authored source for script
                tools, recorded in place of the source.
            discard: Called once when ``call`` never started (a refusal,
                invalid input, a cancelled wait for admission, an exception
                from the admission hook) to release what the caller prepared
                for it.
            input_echo: Builds the ``details.input`` of a ValidationError
                from ``request_kwargs``; None echoes them as given.

        Returns:
            The payload as one JSON content block, plus an image block when it
            carries one.
        """
        # The rate limit keys on the session, which the caller cannot choose;
        # the usage record names the agent the way list_isaac_instances shows
        # it, by its claim's label when the session holds one.
        agent_id = self._resolve_agent_id(None)
        call_started = False
        entered = time.monotonic()
        started: Optional[float] = None

        async def run(request: Optional[BaseModel]) -> Dict[str, Any]:
            nonlocal call_started
            call_started = True
            return await call(request)

        def record(
            duration_ms: float,
            success: bool,
            error: Optional[str],
            *,
            status: Optional[str] = None,
            error_type: Optional[str] = None,
            reason: Optional[str] = None,
            executed: Optional[Union[bool, str]] = None,
        ) -> None:
            # One request id ties this row to the call's audit row, which this
            # also tells how the call ended (the audit used to say "ok" for
            # every call that did not raise). Neither may fail the call.
            try:
                recorded_agent = self._recorded_agent_id(agent_id)
                self.usage_tracker.record(
                    tool_name,
                    duration_ms,
                    success,
                    params=params,
                    error=error,
                    agent_id=recorded_agent,
                    script_sha256=script_sha256,
                    session=agent_id,
                    request_id=current_request_id(),
                    cancelled=status == "cancelled",
                    executed=executed,
                )
                note_call_outcome(
                    # The name the call is recorded under: <unknown tool> for
                    # a made-up one, as in the usage row.
                    tool=tool_name,
                    status=status or ("ok" if success else "error"),
                    error_type=None if success else error_type,
                    agent_id=recorded_agent,
                    session=agent_id,
                    reason=reason,
                    executed=executed,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("Could not record the %s call: %s", tool_name, exc)

        def refuse(refusal: Refusal) -> ToolResult:
            record(
                0.0,
                False,
                refusal.usage_error,
                error_type=refusal.payload.get("error_type"),
                reason=refusal.usage_error,
            )
            return self._as_text_result(refusal.payload)

        try:
            if self._is_paging_continuation(tool_name, params):
                rate_error = self._check_rate_limit(tool_name, agent_id, continuation=True)
            else:
                rate_error = self._check_rate_limit(tool_name, agent_id)
            if rate_error is not None:
                return refuse(Refusal(rate_error, "rate_limited"))
            async with admission(tool_name) as refusal:
                if refusal is not None:
                    return refuse(refusal)
                started = time.monotonic()
                payload = await self._call_outcome(
                    tool_name, run, request_model, request_kwargs, response_model, input_echo
                )
                # ``success`` decides, as it decides isError: a successful
                # script result may carry the script's own ``error`` key.
                failed = payload.get("success") is False
                if failed:
                    normalize_error_payload(payload)
                error = payload.get("error") if failed else None
                record(
                    (time.monotonic() - started) * 1000,
                    not failed,
                    str(error) if error else None,
                    error_type=payload.get("error_type"),
                )
            # The budget pages a paged tool's result by its declaration.
            with tool_result(tool_name):
                return self._as_text_result(payload)
        except asyncio.CancelledError as cancelled:
            # The client gave up. The call is recorded all the same, saying
            # whether the engine ran it: an Isaac call has kept the instance
            # until the engine finished (_exec_isaac), so the duration runs to
            # then and the engine's own outcome is known.
            since = started if started is not None else entered
            duration = (time.monotonic() - since) * 1000
            drained = getattr(cancelled, _DRAINED_OUTCOME, None)
            if not call_started:
                record(duration, False, "cancelled_before_start", status="cancelled", executed=False)
            elif isinstance(drained, CancelledOutcome) and drained.payload is not None:
                outcome = drained.payload
                failed = outcome.get("success") is False
                record(
                    duration,
                    not failed,
                    (str(outcome.get("error") or "") or None) if failed else None,
                    status="cancelled",
                    error_type=outcome.get("error_type"),
                    executed=drained.executed,
                )
            else:
                record(duration, False, "cancelled", status="cancelled", executed="unknown")
            raise
        finally:
            # One place for every way a call can end before it starts. A
            # cancelled wait for the instance lock raises CancelledError out
            # of the admission hook, and would otherwise leave Isaac's tool
            # coroutine unawaited, warning when it is collected.
            if not call_started and discard is not None:
                discard()

    async def _call_outcome(
        self,
        tool_name: str,
        call: Callable[[Optional[BaseModel]], Awaitable[Dict[str, Any]]],
        request_model: Optional[Type[BaseModel]],
        request_kwargs: Optional[Mapping[str, Any]],
        response_model: Optional[Type[BaseModel]],
        input_echo: Optional[InputEcho] = None,
    ) -> Dict[str, Any]:
        """Validate the request, run the call and normalise what it returns.

        The middle stages of ``_run_envelope``; its arguments are documented
        there.

        Returns:
            The call's payload with ``success`` set, the ValidationError
            envelope for invalid input, or the error envelope of the exception
            the call (or the response schema) raised.
        """
        request: Optional[BaseModel] = None
        if request_model is not None:
            validated = self._validate_input(request_model, **(request_kwargs or {}))
            if isinstance(validated, dict):
                if input_echo is not None:
                    validated["details"]["input"] = input_echo(request_kwargs or {})
                return validated
            request = validated
        try:
            payload = await call(request)
            if not isinstance(payload, dict):
                # The engine side answered with something no tool returns: not
                # the caller's input (a TypeError would read as ValidationError).
                self.logger.error(
                    "Tool %s returned %s, not a JSON object", tool_name, type(payload).__name__
                )
                return ErrorResponse(
                    error=f"{tool_name} returned {type(payload).__name__}, not a JSON object",
                    error_type=error_types.ENGINE_ERROR,
                ).model_dump()
            apply_success_from_error(payload)
            if payload.get("success") is False and payload.get("error"):
                # An error envelope built by the tool (input validation,
                # not-found, script failure) keeps its own fields; forcing it
                # through the success schema would only replace the message
                # with a pydantic complaint.
                return payload
            if response_model is not None:
                # Constructing the model is the validation; its dump conforms.
                try:
                    return response_model(**payload).model_dump()
                except PydanticValidationError as exc:
                    # The engine answered with something the tool's response
                    # schema does not describe: not the caller's input.
                    self.logger.error("Tool %s returned an unexpected payload: %s", tool_name, exc)
                    return ErrorResponse(
                        error=(
                            f"{tool_name} received a reply that does not match its response "
                            f"schema {response_model.__name__}: "
                            f"{_summarize_validation_errors(_validation_errors(exc))}"
                        ),
                        error_type=error_types.ENGINE_ERROR,
                    ).model_dump()
            return payload
        except Exception as exc:
            return self._exception_payload(tool_name, exc)

    def _exception_payload(self, tool_name: str, exc: Exception) -> Dict[str, Any]:
        """Build the error envelope for an exception that escaped a tool call.

        A sandbox denial keeps the SandboxError envelope every layer reports.
        Any other exception gets the type of the vocabulary it belongs to
        (``error_types.classify_exception``): the ``error_type`` a structured
        exception declares (Blender's AttachmentStale, an error the Blender
        add-on relays), or its family. A library's or Python's exception name
        is no contract, so it is kept only as ``details.exception_type``. The
        envelope keeps the context the exception's notes add and carries its
        ``details`` when it has any.

        Args:
            tool_name: Registered tool name, for the log line.
            exc: The exception.

        Returns:
            The error envelope.
        """
        if isinstance(exc, SandboxDenied):
            return sandbox_error(exc.details, error=exc.error_message)
        message = exception_message(exc)
        self.logger.error(
            "Tool %s failed: %s: %s", tool_name, exception_type_name(exc), message
        )
        return normalize_error_payload(
            ErrorResponse(error=message, **error_types.exception_fields(exc)).model_dump()
        )

    @asynccontextmanager
    async def _isaac_admission(self, tool_name: str) -> AsyncIterator[Optional[Refusal]]:
        """Admit an Isaac call to the instance the request targets.

        With ``isaac_sim.enforce_claims`` on, a mutating tool is refused while
        another agent holds a live claim on the instance. Then the instance's
        lock is taken, bounded by ``server.timeout`` so a caller queued behind
        a long step learns the instance is busy instead of waiting until its
        own client gives up, which would read as "unreachable". An admitted
        call holds the lock until it has finished, and the caller's binding
        heartbeat is refreshed before the lock is released.

        Args:
            tool_name: Registered tool name, for the read-only check and the
                heartbeat.

        Yields:
            An InstanceClaimed or InstanceBusy refusal, or None once admitted.
        """
        instance_name = self._get_effective_instance_name()
        if self.settings.isaac_sim.enforce_claims and not await self._tool_is_read_only(tool_name):
            holder = self._foreign_claim(instance_name, self._request_agent_id())
            if holder is not None:
                yield Refusal(self._claimed_error(instance_name, holder), "instance_claimed")
                return
        lock = self._get_instance_lock(instance_name)
        if not await self._acquire_instance_lock(lock):
            yield Refusal(
                ErrorResponse(
                    error=(
                        f"Isaac instance {instance_name!r} is busy with another "
                        "call. It is running, not unreachable — retry shortly."
                    ),
                    error_type=error_types.INSTANCE_BUSY,
                ).model_dump(),
                "instance_busy",
            )
            return
        try:
            yield None
            self._heartbeat_active_binding(tool_name)
        finally:
            lock.release()

    async def _acquire_instance_lock(self, lock: asyncio.Lock) -> bool:
        """Wait up to ``server.timeout`` for an Isaac instance's lock.

        Returns:
            True once the lock is held, False when it stayed taken.
        """
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self._instance_lock_timeout)
        except asyncio.TimeoutError:
            return False
        return True

    def _heartbeat_active_binding(self, tool_name: str) -> None:
        """Mark the request's active binding alive after a call made through it.

        Best effort, like the session records it refreshes. The heartbeat runs
        as the admission hook exits, after the call has run and been recorded,
        so a failure here (a session directory that cannot be created or
        read) is logged; raising would escape the envelope and replace the
        call's result with an unhandled error.
        """
        binding = self._get_active_binding()
        if binding is None:
            return
        try:
            self.session_manager.get_instance_session(binding.port).heartbeat(
                binding.agent_id, tool_name
            )
        except Exception as exc:
            self.logger.warning(
                "Could not refresh the claim heartbeat for agent %r on port %s after %s: %s",
                binding.agent_id,
                binding.port,
                tool_name,
                exc,
            )
            return
        binding.last_heartbeat = time.time()

    async def _exec_isaac(
        self,
        tool_name: str,
        coro: Coroutine[Any, Any, Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
        script_sha256: Optional[str] = None,
        *,
        bypass_instance_lock: bool = False,
    ) -> ToolResult:
        """Run an Isaac Sim tool coroutine through the shared envelope.

        Isaac's admission hook (claim enforcement, the instance lock and the
        binding heartbeat) gates the call. ``bypass_instance_lock`` leaves it
        out for a tool that has to reach an instance another call is holding:
        ``interrupt_isaac_script``, which stops that call, and ``ping_isaac``.

        Args:
            tool_name: Registered tool name.
            coro: The coroutine an IsaacTools method returned; closed unawaited
                when the call never runs (refused, cancelled while it waits
                for the instance lock, or stopped by an admission error).
            params: Call parameters worth keeping in the usage log.
            script_sha256: Digest of the agent-authored source for script tools.
            bypass_instance_lock: Skip the claim check and the instance lock.

        Returns:
            The payload as one JSON content block, plus an image block when it
            carries one.

        A call the client cancels keeps the instance until the engine is done
        with it: cancelling the request only closes the socket, and Isaac Sim
        goes on running the script, so releasing the lock then let the next
        call in while the first was still being served. On a cancel the
        bridge is asked to interrupt the script (``interrupt_isaac_script``),
        the call is awaited to its end, bounded by the transport's own
        timeout, and only then is the lock released and the cancellation
        recorded and passed on.
        """

        async def call(_request: Optional[BaseModel]) -> Dict[str, Any]:
            if bypass_instance_lock:
                return await coro
            client = self._get_request_isaac_client()
            task = asyncio.ensure_future(coro)
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError as cancelled:
                # The same exception goes on; it carries what the engine did
                # to the usage and audit records (_run_envelope).
                setattr(
                    cancelled,
                    _DRAINED_OUTCOME,
                    await self._drain_cancelled_isaac_call(tool_name, task, client),
                )
                raise

        return await self._run_envelope(
            tool_name,
            call,
            admission=_admit_every_call if bypass_instance_lock else self._isaac_admission,
            params=params,
            script_sha256=script_sha256,
            discard=coro.close,
        )

    async def _drain_cancelled_isaac_call(
        self, tool_name: str, task: "asyncio.Future[Dict[str, Any]]", client: Any
    ) -> "CancelledOutcome":
        """Wait, shielded from the cancellation, for a cancelled Isaac call to finish.

        Asks the bridge to interrupt the running script first, so a coroutine
        script stops at its next await instead of running to its end. The
        wait is bounded by the client's own socket deadline; a further cancel
        of the waiting request does not cut it short.

        Returns:
            What the engine did: its payload, normalised like any other, and
            whether it ran the operation (``"unknown"`` when the call ended
            without an answer, or with a connection failure or timeout that
            leaves it open).
        """
        if not task.done():
            await self._wait_for_cancelled_isaac_call(tool_name, task, client)
        if task.cancelled() or task.exception() is not None:
            if not task.cancelled():
                self.logger.debug(
                    "The cancelled %s ended with %s", tool_name, type(task.exception()).__name__
                )
            return CancelledOutcome("unknown", None)
        payload = task.result()
        if not isinstance(payload, dict):
            return CancelledOutcome("unknown", None)
        apply_success_from_error(payload)
        if payload.get("success") is False:
            normalize_error_payload(payload)
            if payload.get("error_type") in (
                error_types.CONNECTION_ERROR,
                error_types.TIMEOUT_ERROR,
            ):
                return CancelledOutcome("unknown", payload)
        return CancelledOutcome(True, payload)

    async def _wait_for_cancelled_isaac_call(
        self, tool_name: str, task: "asyncio.Future[Dict[str, Any]]", client: Any
    ) -> None:
        """Interrupt the cancelled call's script and wait for it, shielded."""
        self.logger.info(
            "%s was cancelled by the client; holding the Isaac instance until it ends", tool_name
        )
        with anyio.CancelScope(shield=True):
            if getattr(client, "bridge_enabled", False):
                try:
                    await asyncio.wait_for(
                        client.interrupt_bridge_script(), timeout=_INTERRUPT_TIMEOUT_SECONDS
                    )
                except Exception as exc:
                    self.logger.debug("Could not interrupt the cancelled %s: %s", tool_name, exc)
            while not task.done():
                try:
                    await asyncio.wait({task})
                except asyncio.CancelledError:
                    # A cancellation anyio's shield does not cover (a bare
                    # task.cancel()); the engine is still busy, keep waiting.
                    continue

    async def _exec_payload(
        self,
        tool_name: str,
        payload: Union[Dict[str, Any], Callable[[], Awaitable[Dict[str, Any]]]],
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Run a tool whose result the server computes itself through the shared envelope.

        For the server's own tools (instance routing, claims, usage stats) and
        for refusals decided before a backend is reached (an Isaac path
        outside the sandbox), so they are rate limited, recorded and sent
        like every other call.

        Args:
            tool_name: Registered tool name.
            payload: The payload, or a coroutine function computing it inside
                the envelope (an exception it raises becomes an error
                envelope).
            params: Call parameters worth keeping in the usage log.

        Returns:
            The payload as one JSON content block.
        """

        async def call(_request: Optional[BaseModel]) -> Dict[str, Any]:
            if callable(payload):
                return await payload()
            return payload

        return await self._run_envelope(tool_name, call, params=params)

    def _sandbox_denial(
        self, path_str: Optional[str], *, write: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Return the SandboxError payload for ``path_str``, or None when allowed.

        A fast-fail at the MCP boundary; the authoritative check lives in the
        tools and session layers, below both this and the CLI.

        Args:
            path_str: Path or URL supplied by the caller, or None when the tool
                received no path to police.
            write: Whether the tool writes to the location.

        Returns:
            The error envelope naming the allowed roots and URL schemes, or None.
        """
        return self._path_policy.denial(path_str, write=write)

    def _validate_input(
        self, model: Type[BaseModel], **kwargs: Any
    ) -> Union[BaseModel, Dict[str, Any]]:
        """Validate tool arguments against ``model``.

        Args:
            model: The request model.
            **kwargs: The tool arguments.

        Returns:
            The model instance, or the ValidationError envelope naming the
            input when the arguments do not fit it.
        """
        try:
            return model(**kwargs)
        except Exception as e:
            return validation_error_payload(e, kwargs)

    async def _exec_backend(
        self,
        tool_name: str,
        adapter: Any,
        adapter_label: str,
        response_model: Type[BaseModel],
        call: BackendCall,
        params: Optional[Dict[str, Any]] = None,
        *,
        request_model: Optional[Type[BaseModel]] = None,
        request_kwargs: Optional[Mapping[str, Any]] = None,
        input_echo: Optional[InputEcho] = None,
    ) -> ToolResult:
        """
        Run a USD, Blender or Unreal tool through the shared envelope.

        The envelope (``_run_envelope``) carries what every tool owes the
        operator and the caller: the per-agent rate limit, request
        validation, exception wrapping, success normalisation, the response
        schema, the usage record, the result budget and single transmission.
        What is particular to these backends is the call itself
        (``_run_backend_call``): the availability check and the session scope.

        Args:
            tool_name: Registered tool name, used for rate limiting and logs.
            adapter: Runtime adapter, or None when the backend is absent.
            adapter_label: Human-readable backend name for the error message.
            response_model: Schema a successful payload is validated against.
            call: Receives an open session, followed by the validated request
                when ``request_model`` is given, and returns the payload, or an
                error envelope (``success`` False with ``error``) that is
                passed through as is. May be sync or async.
            params: Call parameters worth keeping in the usage log.
            request_model: Model ``request_kwargs`` are validated against
                inside the envelope, so invalid input is rate limited and
                recorded like any other failed call. Without it, parse tool
                input inside ``call``, not before it.
            request_kwargs: The tool arguments to validate.
            input_echo: Builds the ``details.input`` a ValidationError echoes
                from ``request_kwargs``; None echoes them as given.

        Returns:
            The payload as one JSON content block, plus an image block when
            it carries one.
        """

        async def run(request: Optional[BaseModel]) -> Dict[str, Any]:
            return await self._run_backend_call(adapter, adapter_label, call, request)

        return await self._run_envelope(
            tool_name,
            run,
            request_model=request_model,
            request_kwargs=request_kwargs,
            response_model=response_model,
            params=params,
            input_echo=input_echo,
        )

    async def _run_backend_call(
        self,
        adapter: Any,
        adapter_label: str,
        call: BackendCall,
        request: Optional[BaseModel] = None,
    ) -> Dict[str, Any]:
        """Open a session and run ``call`` in it.

        An exception propagates to the envelope, which reports it.

        Args:
            adapter: Runtime adapter, or None when the backend is absent.
            adapter_label: Human-readable backend name for the error message.
            call: Receives the open session (and ``request``, when given) and
                returns the payload.
            request: The validated request, passed to ``call`` after the
                session.

        Returns:
            The payload, or the error envelope of an absent backend.
        """
        if adapter is None or not adapter.is_available():
            return ErrorResponse(
                error=f"{adapter_label} runtime not available",
                error_type=error_types.CONNECTION_ERROR,
            ).model_dump()

        arguments = () if request is None else (request,)
        with adapter.create_session() as session:
            # Unreal sessions are async, Blender sessions are sync; accept
            # both so one envelope serves every backend.
            if getattr(adapter, "session_blocks_io", False):
                # The adapter declares its session only waits on I/O (never
                # bpy), so keep other MCP requests responsive meanwhile.
                payload = await asyncio.to_thread(call, session, *arguments)
            else:
                payload = call(session, *arguments)
            if inspect.isawaitable(payload):
                payload = await payload
        return payload

    def _as_text_result(self, payload: Dict[str, Any]) -> ToolResult:
        """Return ``payload`` within the result budget, sent once.

        Returning a plain dict makes FastMCP emit the payload twice — once as
        JSON text and again as ``structuredContent`` — and both copies land in
        the caller's context window. A content-only ToolResult is the one shape
        that sends it once; ``output_schema=None`` alone does not (it drops the
        schema from the listing but keeps the duplicate).

        A payload carrying ``image_base64`` (viewport captures, asset
        thumbnails) has the image lifted into an ``ImageContent`` block the
        client can render; inside the JSON it is opaque text at roughly one
        token per three bytes. The result budget is applied here, after the
        lift, so every envelope is covered by one rule and an image never
        counts against the text it accompanies.

        A payload with ``success: false`` is sent with ``isError: true``, and
        with an ``error_type`` from the vocabulary (``normalize_error_payload``),
        whichever path built it.

        Args:
            payload: The tool's result dict.

        Returns:
            One JSON text block, preceded by an image block when present.
        """
        if payload.get("success") is False:
            normalize_error_payload(payload)
        content: List[Any] = []
        image = payload.get("image_base64")
        if "image_base64" in payload:
            # The transport keys never reach the client: a present image moves
            # to its own block, an absent one (a null a response model filled
            # in) would only be noise in the record.
            payload = {
                key: value
                for key, value in payload.items()
                if key not in ("image_base64", "encoding")
            }
        if isinstance(image, str) and image:
            payload["image_attached"] = True
            image_format = _sniff_image_format(image) or str(
                payload.get("format", "png")
            ).lower()
            mime_type = (
                "image/jpeg" if image_format in ("jpg", "jpeg") else f"image/{image_format}"
            )
            content.append(ImageContent(type="image", data=image, mime_type=mime_type))
        text = encode_result_budget(payload)
        content.append(TextContent(type="text", text=text))
        # MCP's isError is what clients branch on before they parse the JSON:
        # every failed call is one, not only the ones FastMCP refuses itself.
        return ToolResult(content=content, is_error=payload.get("success") is False)

    def _script_tool(self, **tool_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register an arbitrary-code tool, or leave it off the surface.

        ``security.allow_script_execution`` is the operator's switch for the
        agent-authored code path. Granular tools are generated scripts and stay
        registered either way. Script execution and generic Unreal function/batch
        dispatch tools go through here.

        Args:
            **tool_kwargs: Keyword arguments for ``FastMCP.tool``.

        Returns:
            The FastMCP registration decorator, or an identity decorator when
            script execution is disabled.
        """
        if self.settings.security.allow_script_execution:
            return self.mcp.tool(**tool_kwargs)

        def unregistered(func: Callable[..., Any]) -> Callable[..., Any]:
            self._hidden_script_tools.add(str(tool_kwargs.get("name") or func.__name__))
            return func

        return unregistered

    @staticmethod
    def _script_execution_disabled(tool_name: str) -> Dict[str, Any]:
        """The ScriptExecutionDisabled envelope for a call to a hidden arbitrary-code tool."""
        return ErrorResponse(
            error=(
                f"{tool_name} is not available: arbitrary script execution is disabled by "
                "security.allow_script_execution. Use the granular tools, or set "
                "SECURITY__ALLOW_SCRIPT_EXECUTION=true to enable it."
            ),
            error_type=error_types.SCRIPT_EXECUTION_DISABLED,
            details={"tool": tool_name},
        ).model_dump()

    def _tool_annotations(
        self,
        read_only: bool,
        idempotent: bool,
        open_world: bool,
        destructive: bool = False,
    ) -> Optional[Any]:
        """Build the MCP annotations for one tool.

        Every hint the call site computes is preserved. readOnlyHint is always
        emitted because auto-approval policies key on it, and so is
        destructiveHint on every tool that is not read-only: the MCP spec's
        default for a missing one is true, so leaving it off an additive tool
        told spec-following clients it was destructive. A read-only tool
        leaves it off (the spec gives it meaning only for a tool that is not
        read-only), as it leaves off a hint equal to the spec's default
        (idempotent false, open world true).

        Args:
            read_only: The tool does not modify its environment.
            idempotent: Repeating the call with the same arguments has no
                additional effect.
            open_world: The tool talks to an external process or the network.
            destructive: The tool overwrites or deletes existing state or files.

        Returns:
            A ``ToolAnnotations`` instance, or the plain dict when FastMCP does
            not expose the model.
        """
        annotations: Dict[str, Any] = {"readOnlyHint": read_only}
        if not read_only:
            annotations["destructiveHint"] = destructive
        if idempotent:
            annotations["idempotentHint"] = True
        if not open_world:
            annotations["openWorldHint"] = False
        if ToolAnnotations:
            return ToolAnnotations(**annotations)
        return annotations

    def _task_optional(self) -> Optional[Any]:
        """
        Return a TaskConfig for optional background task support.

        FastMCP 4 runs background tasks through its tasks extension and
        refuses to start a server that has a task-enabled tool but no such
        extension, so a tool opts in only when ``_register_tasks_extension``
        registered it.
        """
        if TaskConfig and self._tasks_enabled:
            return TaskConfig(mode="optional")
        return None

    def _register_tasks_extension(self) -> bool:
        """Register FastMCP's background tasks extension when it is installed.

        The extension (``io.modelcontextprotocol/tasks``) ships in the
        optional ``fastmcp-tasks`` package (``fastmcp[tasks]``). Without it
        every tool runs in the foreground, as on FastMCP 3 without the docket
        runtime; a broken install is logged and skipped rather than stopping
        the server.

        Returns:
            True when the extension is registered.
        """
        if not TaskConfig:
            return False
        try:
            from fastmcp_tasks import TasksExtension
        except ImportError:
            return False
        try:
            self.mcp.add_extension(TasksExtension())
        except Exception as exc:
            self.logger.warning(
                "Background tasks disabled: fastmcp-tasks did not register: %s", exc
            )
            return False
        return True

    def _register_isaac_resources(self) -> None:
        """Register the Isaac Sim scripting references as MCP resources.

        Registered with the Isaac tools, which they document. The documents
        ship inside the package (``simul/resources``) so a wheel install
        serves them the same way an editable checkout does.
        """
        register_resource = getattr(self.mcp, "resource", None)
        if not callable(register_resource):
            self.logger.debug(
                "FastMCP resource API unavailable; skipping resource registration"
            )
            return

        def _read_packaged_doc(*parts: str) -> str:
            """Return a packaged Markdown document, or a note naming what is missing."""
            document = resource(*parts)
            if document.is_file():
                return document.read_text(encoding="utf-8")
            return f"{'/'.join(parts)} is missing from the simul package."

        @register_resource(
            "simul://isaac-sim/skills",
            name="Isaac Sim Scripting Skills",
            description=(
                "Isaac Sim 5.1 / 6.0 scripting reference: API patterns, "
                "namespace migration notes, and quick-reference table. "
                "Only consult this when no granular tool exists for your task."
            ),
        )
        def isaac_sim_skills() -> str:
            return _read_packaged_doc("skills.md")

        @register_resource(
            "simul://isaac-sim/api/core",
            name="Isaac Sim Core API",
            description=(
                "Isaac Sim 5.1 SimulationContext, PhysicsContext, Articulation, RigidPrim, "
                "XFormPrim reference (deprecated on 6.0), with the isaacsim.core.experimental "
                "equivalents to use on 6.0."
            ),
        )
        def api_core() -> str:
            return _read_packaged_doc("docs", "api", "core.md")

        @register_resource(
            "simul://isaac-sim/api/sensors",
            name="Isaac Sim Sensors API",
            description="Camera, IMU, Contact, LiDAR (PhysX/RTX), Proximity sensor reference.",
        )
        def api_sensors() -> str:
            return _read_packaged_doc("docs", "api", "sensors.md")

        @register_resource(
            "simul://isaac-sim/api/physics",
            name="Isaac Sim Physics API",
            description="PhysX interface, tensor API, collision queries, CCT, vehicle physics reference.",
        )
        def api_physics() -> str:
            return _read_packaged_doc("docs", "api", "physics.md")

        @register_resource(
            "simul://isaac-sim/api/replicator",
            name="Isaac Sim Replicator API",
            description="Annotators, Writers, Orchestrator, domain randomization reference.",
        )
        def api_replicator() -> str:
            return _read_packaged_doc("docs", "api", "replicator.md")

        @register_resource(
            "simul://isaac-sim/api/robots",
            name="Isaac Sim Robots API",
            description="Manipulators, grippers, IK, motion planning, wheeled robots reference.",
        )
        def api_robots() -> str:
            return _read_packaged_doc("docs", "api", "robots.md")

        @register_resource(
            "simul://isaac-sim/api/rendering",
            name="Isaac Sim Rendering API",
            description="Viewport, HydraTexture, RTX post-processing, capture reference.",
        )
        def api_rendering() -> str:
            return _read_packaged_doc("docs", "api", "rendering.md")

        @register_resource(
            "simul://isaac-sim/api/assets",
            name="Isaac Sim Assets API",
            description="URDF/MJCF import, Cloner, OmniGraph nodes reference.",
        )
        def api_assets() -> str:
            return _read_packaged_doc("docs", "api", "assets.md")

    def _switch_active_instance(self, name: str) -> None:
        """
        Switch the default Isaac Sim instance by name.

        This only affects requests that do not have a session-scoped route.

        Args:
            name: Instance name key in ``_isaac_clients``.
        """
        client = self._isaac_clients[name]
        self._active_instance = name
        self.client = client

    def _bridge_port_for_socket(self, socket_port: int) -> int:
        """Derive the bridge port for an Isaac instance from its socket port."""
        return self.isaac_adapter.bridge_port_for_socket(socket_port)

    def _socket_port_for_bridge(self, bridge_port: int) -> int:
        """Derive the VS Code socket port for an instance from its bridge port."""
        return self.isaac_adapter.socket_port_for_bridge(bridge_port)

    def _build_isaac_client(
        self,
        *,
        socket_host: str,
        socket_port: int,
        socket_timeout: float,
        bridge_enabled: bool,
        bridge_host: Optional[str] = None,
        bridge_port: Optional[int] = None,
        bridge_timeout: Optional[float] = None,
        bridge_fallback_to_vscode: Optional[bool] = None,
        bridge_socket_path: Optional[str] = None,
        socket_protocol: Optional[str] = None,
        socket_auth_token: Optional[str] = None,
    ) -> IsaacSocketClient:
        """Create one bridge-aware Isaac client from default or per-instance config."""
        return self.isaac_adapter.build_client(
            socket_host=socket_host,
            socket_port=socket_port,
            socket_timeout=socket_timeout,
            bridge_enabled=bridge_enabled,
            bridge_host=bridge_host,
            bridge_port=bridge_port,
            bridge_timeout=bridge_timeout,
            bridge_fallback_to_vscode=bridge_fallback_to_vscode,
            bridge_socket_path=bridge_socket_path,
            socket_protocol=socket_protocol,
            socket_auth_token=socket_auth_token,
        )

    async def _discover_from_files(self) -> Dict[str, IsaacSocketClient]:
        """
        Discover Isaac Sim instances from bridge discovery files.

        Each running bridge extension writes a JSON file to the discovery
        directory containing its PID, host, and actual bound port.
        Stale files (dead PIDs) are cleaned up automatically.
        """
        discovery_dir = self.settings.isaac_sim.discovery_dir
        if not os.path.isdir(discovery_dir):
            return {}
        # Every entry below is trusted, so the directory has to be ours alone:
        # another local user who can write here can point the server at a
        # loopback port they own and read every script we send.
        dir_problem = DiscoveryDir(discovery_dir).problem()
        if dir_problem is not None:
            self.logger.warning(
                "Skipping Isaac discovery files: %s. Make it private (chmod 700) or point "
                "isaac_sim.discovery_dir elsewhere.",
                dir_problem,
            )
            return {}

        existing_ports: set[int] = set()
        for c in self._isaac_clients.values():
            existing_ports.add(c._port)  # vscode port
            if c._bridge_configured and c._bridge_port is not None:
                existing_ports.add(c._bridge_port)  # bridge port

        candidates: List[tuple[str, IsaacSocketClient]] = []
        for filename in os.listdir(discovery_dir):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(discovery_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.loads(f.read())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                # Valid JSON that is not a discovery record (a list, a bare
                # string) must not take the whole listing down with it.
                self.logger.warning(
                    "Skipping Isaac discovery file %s: expected a JSON object, got %s",
                    filepath,
                    type(data).__name__,
                )
                continue

            pid = data.get("pid")
            host = data.get("host", "127.0.0.1")
            port = data.get("port")
            vscode_port = data.get("vscode_port")
            socket_path = data.get("socket_path")

            # The discovery dir is the trust boundary for sockets, exactly as
            # loopback is for TCP: a hostile or corrupted entry must not point
            # the client at an arbitrary socket elsewhere on the filesystem.
            if socket_path is not None:
                resolved = os.path.realpath(str(socket_path))
                boundary = os.path.realpath(discovery_dir) + os.sep
                if not resolved.startswith(boundary):
                    # A containerised bridge advertises its own mount point
                    # (/tmp/simul/...), not the host's. The socket must
                    # live in the discovery dir anyway, so try its basename
                    # inside the local dir — inside the boundary by
                    # construction.
                    resolved = os.path.realpath(
                        os.path.join(discovery_dir, os.path.basename(str(socket_path)))
                    )
                if not resolved.startswith(boundary) or not os.path.exists(resolved):
                    socket_path = None
                elif len(os.fsencode(resolved)) >= _UNIX_SOCKET_PATH_MAX:
                    # The host's discovery dir can be deeper than the
                    # container's, and connect(2) rejects a path sun_path
                    # cannot hold; TCP still reaches the bridge.
                    logger.warning(
                        "Bridge socket %s is too long for AF_UNIX; using TCP instead", resolved
                    )
                    socket_path = None
                else:
                    socket_path = resolved

            # Only trust loopback addresses from discovery files
            if host not in ("127.0.0.1", "::1", "localhost"):
                continue

            if not isinstance(port, int) or port in existing_ports:
                continue
            if vscode_port is not None and not isinstance(vscode_port, int):
                continue

            if isinstance(pid, int) and not DiscoveryDir.pid_alive(pid):
                # Process is dead -- clean up stale file
                try:
                    os.remove(filepath)
                except OSError:
                    pass
                continue

            client = self._build_isaac_client(
                socket_host=host,
                socket_port=(
                    vscode_port
                    if isinstance(vscode_port, int)
                    else self._socket_port_for_bridge(port)
                ),
                socket_timeout=min(self.settings.isaac_sim.socket_timeout, 3.0),
                bridge_enabled=True,
                bridge_host=host,
                bridge_port=port,
                bridge_timeout=min(self.settings.isaac_sim.bridge_timeout, 3.0),
                bridge_fallback_to_vscode=self.settings.isaac_sim.bridge_fallback_to_vscode,
                bridge_socket_path=socket_path,
            )

            candidates.append((f"isaac-{port}", client))

        if not candidates:
            return {}

        # Verify reachability for every candidate at once. Each has its own
        # socket, so there is nothing to serialise, and a stale discovery file
        # would otherwise burn its full timeout before the next is even tried.
        alive = await asyncio.gather(*(client.ping() for _, client in candidates))
        return {
            name: client
            for (name, client), reachable in zip(candidates, alive)
            if reachable
        }

    async def _scan_isaac_instances(self) -> Dict[str, IsaacSocketClient]:
        """
        Scan the configured port range for running Isaac Sim instances.

        Returns any newly discovered instances not already in the registry.
        Existing named instances are never overwritten.  Phase 1 discovers
        instances via discovery files; Phase 2 fills gaps with a port scan.
        All port-scan candidate pings run concurrently via ``asyncio.gather``
        to avoid sequential timeouts.

        Ports that are bridge endpoints — the configured default, every
        per-instance override, and any bridge advertised by a discovery file
        or already registered — are excluded from the scan. Probing one with
        the stock-socket protocol never finds an instance: the bridge waits
        for a length prefix that never comes, and the probe would only return
        once its read deadline expired.
        """
        # Phase 1: fast discovery via files
        file_discovered = await self._discover_from_files()

        # Phase 2: port scan for instances without discovery files
        scan_start = self.settings.isaac_sim.scan_port_start
        scan_end = self.settings.isaac_sim.scan_port_end
        host = self.settings.isaac_sim.socket_host
        timeout = self.settings.isaac_sim.socket_timeout
        existing_ports: set[int] = {
            c._port for c in self._isaac_clients.values() if c._host == host
        }
        existing_ports.update(c._port for c in file_discovered.values())
        bridge_ports: set[int] = {self.settings.isaac_sim.bridge_port}
        bridge_ports.update(
            inst.bridge_port
            for inst in self.settings.isaac_sim.instances
            if inst.bridge_port is not None
        )
        for known in (*self._isaac_clients.values(), *file_discovered.values()):
            if known._bridge_port is not None:
                bridge_ports.add(known._bridge_port)

        candidates: Dict[int, IsaacSocketClient] = {}
        for port in range(scan_start, scan_end):
            if port in existing_ports or port in bridge_ports:
                continue
            candidates[port] = self._build_isaac_client(
                socket_host=host,
                socket_port=port,
                socket_timeout=min(timeout, 3.0),
                bridge_enabled=self.settings.isaac_sim.bridge_enabled,
                bridge_host=self.settings.isaac_sim.bridge_host,
                bridge_port=self._bridge_port_for_socket(port),
                bridge_timeout=min(self.settings.isaac_sim.bridge_timeout, 3.0),
                bridge_fallback_to_vscode=self.settings.isaac_sim.bridge_fallback_to_vscode,
            )

        if not candidates:
            return file_discovered

        ping_results = await asyncio.gather(
            *(client.ping() for client in candidates.values()),
            return_exceptions=True,
        )

        discovered: Dict[str, IsaacSocketClient] = {}
        for (port, client), result in zip(candidates.items(), ping_results):
            if result is True:
                discovered[f"isaac-{port}"] = client

        # Merge: file discovery takes priority
        file_discovered.update(discovered)
        return file_discovered

    async def _get_instance_brief(
        self, name: str, client: IsaacSocketClient
    ) -> Dict[str, Any]:
        """
        Get brief stage info from an Isaac Sim instance.

        Args:
            name: Instance identifier.
            client: Socket client for this instance.

        Returns:
            Dict with instance info fields.
        """
        active_instance = self._get_effective_instance_name()
        info: Dict[str, Any] = {
            "name": name,
            "host": client._host,
            "port": client._port,
            "bridge_address": client.bridge_address,
            "vscode_address": client.vscode_address,
            "reachable": False,
            "active": name == active_instance,
            "stage_url": None,
            "up_axis": None,
            "prim_count": None,
            "is_playing": None,
        }
        try:
            if client.bridge_enabled:
                # The listing reports a prim count only incidentally, and
                # producing one costs a full stage traversal inside Kit for
                # every instance enumerated. Ask stage info to skip it; callers
                # that want the number have get_isaac_stage_info.
                stage_info = await client.bridge_request(
                    "get_stage_info", {"include_prim_count": False}
                )
                sim_state = await client.bridge_request("get_simulation_state", {})
                if stage_info.get("status") == "ok":
                    payload = stage_info.get("payload", {})
                    info["reachable"] = True
                    info["stage_url"] = payload.get("stage_url")
                    info["up_axis"] = payload.get("up_axis")
                    info["prim_count"] = payload.get("total_prims")
                if sim_state.get("status") == "ok":
                    payload = sim_state.get("payload", {})
                    info["reachable"] = True
                    info["is_playing"] = payload.get("is_playing")
                if info["reachable"]:
                    return info

            if not client.fallback_to_vscode:
                return info

            result = await client.execute_vscode_only(
                "import json, omni.usd, omni.timeline\n"
                "ctx = omni.usd.get_context()\n"
                "stage = ctx.get_stage()\n"
                "tl = omni.timeline.get_timeline_interface()\n"
                "from pxr import UsdGeom\n"
                "print(json.dumps({\n"
                "    'stage_url': ctx.get_stage_url(),\n"
                "    'up_axis': UsdGeom.GetStageUpAxis(stage) if stage else None,\n"
                # Skipped for the same reason as the bridge path: a per-instance
                # stage traversal is far too expensive for a listing field.
                "    'prim_count': None,\n"
                "    'is_playing': tl.is_playing(),\n"
                "}))\n"
            )
            if result.success:
                info["reachable"] = True
                data = json.loads(result.output)
                info["stage_url"] = data.get("stage_url")
                info["up_axis"] = data.get("up_axis")
                info["prim_count"] = data.get("prim_count")
                info["is_playing"] = data.get("is_playing")
            else:
                info["reachable"] = True
        except (ConnectionRefusedError, TimeoutError, OSError, ValueError):
            info["reachable"] = False
        except (json.JSONDecodeError, KeyError):
            info["reachable"] = True
        return info

    def _skip_reason(self, name: str) -> Optional[str]:
        """Return why backend *name* registers no tools whatever its runtime, or None.

        Either the ``backends`` selection left it out or its spec's gate
        (Isaac Sim's platform check) holds it back.
        """
        return backend_skip_reason(name, self.settings, self._backends)

    def _backend_enabled(self, name: str) -> bool:
        """Return True when *name* is selected and no gate holds its tools back."""
        return self._skip_reason(name) is None

    def _registers(self, name: str) -> bool:
        """Return True when backend *name* gets its tools registered.

        An enabled backend without an adapter, or whose runtime is not
        available in this process, registers nothing: its tools would fail
        on every call. ``require_available=False`` lists them anyway.
        """
        if not self._backend_enabled(name):
            return False
        if not self._require_available:
            return True
        adapter = self._adapters[name]
        return adapter is not None and adapter.is_available()

    def _register_tools(self) -> None:
        """Register the tools of the backends ``_registered_backends`` names, then the server's own.

        A selected backend that a gate holds back (Isaac Sim on an OS it does
        not run on) is logged with the reason, so the skip is not silent.
        """
        for spec in BACKENDS:
            if spec.name not in self._registered_backends:
                reason = self._skip_reason(spec.name)
                if reason is not None and reason != NOT_SELECTED:
                    self.logger.info("%s tools skipped: %s", spec.label, reason)
                continue
            before = self._registered_tool_names()
            spec.register_tools(self)
            for name in self._registered_tool_names() - before:
                self._tool_backends[name] = spec.name

        # Usage statistics (always available)
        register_stats_tools(self)

        self.logger.info(f"Registered {len(self._registered_tool_names())} MCP tools")

    async def shutdown(self) -> None:
        """Release server resources on shutdown."""
        for adapter in self._adapters.values():
            if adapter is not None:
                adapter.close()
        self._isaac_clients.clear()
        self._rate_limiters.clear()
        self.usage_tracker._stats.clear()
        self.usage_tracker._session_stats.clear()
        self.usage_tracker._recent.clear()
        logger.info("Simul 3D MCP Server shut down")

    async def run(self, transport: str = "stdio") -> None:
        """
        Run the MCP server.

        Args:
            transport: One of ``TRANSPORTS``: ``stdio`` for a client that
                spawns the server, ``http`` (streamable HTTP) or ``sse`` for a
                network listener on ``server.host``:``server.port``.

        Raises:
            ValueError: When ``transport`` is not one of ``TRANSPORTS``.
        """
        try:
            self.logger.info(f"Starting Simul 3D MCP Server with {transport} transport")

            # No FastMCP banner: it checks PyPI for a newer FastMCP on every
            # start (+2 s when PyPI is unreachable) and caches the answer in
            # FastMCP's home. The CLI prints simul's own banner.
            if transport == "stdio":
                await self.mcp.run_async(transport="stdio", show_banner=False)
            elif transport in NETWORK_TRANSPORTS:
                from .http_security import resolve_http_security

                # Resolved before the listener opens: a public bind without
                # --allow-public or a bad token file stops here (see http_security).
                http_security = resolve_http_security(self.settings.server)
                for warning in http_security.warnings:
                    self.logger.warning(warning)
                auth = http_security.build_auth()
                if auth is not None:
                    self.mcp.auth = auth
                await self.mcp.run_async(
                    transport=cast(Literal["http", "sse"], transport),
                    show_banner=False,
                    host=self.settings.server.host,
                    port=self.settings.server.port,
                    **http_security.run_kwargs(),
                )
            else:
                raise ValueError(
                    f"Unsupported transport: {transport}. Valid: {', '.join(TRANSPORTS)}"
                )

        except Exception as e:
            self.logger.error("Error running MCP server: %s", e)
            raise
        finally:
            await self.shutdown()

    def get_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Report every backend's availability and capability names.

        Availability here means the backend is wired up in this process
        (adapter importable and switched on in settings); it is not a
        liveness probe of the engine, which needs a network round-trip.

        Returns:
            Mapping of every registered backend name to ``{"enabled": bool,
            "available": bool, "capabilities": list[str], "skipped":
            Optional[str]}``. ``enabled`` is whether the backend was selected
            for tool registration and no gate held it back; ``skipped`` says
            why it was not (left out of ``backends``, or Isaac Sim on an OS
            it does not run on) and is None for an enabled backend. A backend
            that is not enabled has no adapter and reports ``available``
            false (Isaac excepted, whose adapter always exists).
        """
        return {
            spec.name: self._adapter_capabilities(spec.name, self._adapters[spec.name])
            for spec in BACKENDS
        }

    def _adapter_capabilities(
        self, backend: str, adapter: Optional[BackendAdapter]
    ) -> Dict[str, Any]:
        """Build one ``get_capabilities`` entry for a backend."""
        available = adapter is not None and adapter.is_available()
        skipped = self._skip_reason(backend)
        return {
            "enabled": skipped is None,
            "available": available,
            "capabilities": sorted(adapter.get_capabilities()) if available and adapter else [],
            "skipped": skipped,
        }


# Convenience functions
def create_server_instance(
    settings: Optional[Settings] = None,
    backends: Optional[Set[str]] = None,
) -> SimulMCPServer:
    """
    Create a Simul 3D MCP Server instance.

    Args:
        settings: Configuration settings
        backends: Backend names to enable (None = all available)

    Returns:
        SimulMCPServer instance
    """
    return SimulMCPServer(settings, backends=backends)


async def start_mcp_server(
    settings: Optional[Settings] = None,
    transport: str = "stdio",
    backends: Optional[Set[str]] = None,
) -> None:
    """
    Start the Simul 3D MCP Server.

    Args:
        settings: Configuration settings
        transport: Transport type
        backends: Backend names to enable (None = all available)
    """
    server = create_server_instance(settings, backends=backends)
    await server.run(transport)
