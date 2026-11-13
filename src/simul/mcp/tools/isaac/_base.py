"""Transport and policy machinery every Isaac tool mixin builds on."""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ....adapters import IsaacSocketClient, ScriptResult
from ....adapters.isaac_socket_client import BridgeRequestDeliveredError
from ....config import Settings, get_settings
from ....logging import LoggerMixin
from ....utils.paths import PathPolicy
from ... import error_types
from ...registration._helpers import apply_success_from_error
from ...schemas.common import ErrorResponse
from ._shared import (
    ANNOTATION_MARKER,
    HEAVY_SCRIPT_TIMEOUT_SECONDS,
    SCRIPT_ANNOTATIONS,
    logger,
)

#: Types for the failures simul's own scripts print as ``{"error": ...}``
#: without an ``error_type``, by the fixed wording those scripts use; the
#: first pattern that matches the start of the message wins. Anything else
#: ("Failed to ...", an exception's text) stays ``EngineError``.
SCRIPT_FAILURE_PATTERNS: Tuple[Tuple["re.Pattern[str]", str], ...] = (
    (re.compile(r"No stage is (?:currently )?open"), error_types.INVALID_STATE),
    (re.compile(r"No (?:active )?viewport"), error_types.VIEWPORT_UNAVAILABLE),
    (re.compile(r"Refusing "), error_types.REFUSED_OPERATION),
    (re.compile(r"Viewport capture (?:failed|error)"), error_types.CAPTURE_ERROR),
    (re.compile(r"[\w ]{0,60}\bnot available\b"), error_types.UNSUPPORTED_OPERATION),
    (
        re.compile(r"[\w ]{0,60}\bnot found\b|No [\w ]{1,60}\b(?:found|on the stage)\b"),
        error_types.NOT_FOUND_ERROR,
    ),
    (re.compile(r"[\w ]{0,60}\balready (?:exists|applied)\b"), error_types.INVALID_STATE),
    (
        re.compile(r"Invalid |Not a valid |Prim (?:is not|is inside|does not have) |[\w ]{0,60} must be "),
        error_types.VALIDATION_ERROR,
    ),
)


def script_failure_type(message: Any) -> str:
    """Return the vocabulary type of an untyped failure a tool script printed.

    Args:
        message: The script's ``error`` value.

    Returns:
        The type of the first ``SCRIPT_FAILURE_PATTERNS`` entry matching the
        start of the message, else ``EngineError``.
    """
    text = str(message or "").strip()
    for pattern, error_type in SCRIPT_FAILURE_PATTERNS:
        if pattern.match(text):
            return error_type
    return error_types.ENGINE_ERROR


#: How much of the stdout that was not the tool's JSON a reply carries back.
FOREIGN_OUTPUT_REPORT_CHARS = 2000
#: Positions in one line tried as the start of the tool's JSON object.
_MAX_JSON_START_ATTEMPTS = 8


def extract_tool_json(output: str) -> Tuple[Optional[Any], str]:
    """Return the JSON a tool script printed and any other text around it.

    A generated script prints exactly one JSON object, as its last act. While
    it awaits frames, other Python code in Kit (a physics callback, a script
    node, an extension) can print into the same stdout capture: every
    transport redirects the process-wide ``sys.stdout``, and the stock
    sockets do so across the script's awaits. Such text lands before the
    script's final line, or glued to its start when it printed no newline,
    never after it, because the capture ends in the step that prints it.

    Args:
        output: The script's captured stdout.

    Returns:
        ``(value, other_output)``: the parsed JSON, or None when no line ends
        in a JSON object, and the remaining text (empty when the whole output
        was the JSON).
    """
    text = output.strip()
    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        pass
    lines = text.splitlines()
    decoder = json.JSONDecoder()
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index].strip()
        if not line.endswith("}"):
            continue
        start = line.find("{")
        attempts = 0
        while start != -1 and attempts < _MAX_JSON_START_ATTEMPTS:
            attempts += 1
            try:
                value, end = decoder.raw_decode(line, start)
            except json.JSONDecodeError:
                value, end = None, -1
            if end == len(line) and isinstance(value, dict):
                other = lines[:index] + [line[:start]] + lines[index + 1:]
                return value, "\n".join(part for part in other if part.strip()).strip()
            start = line.find("{", start + 1)
    return None, text


def split_annotations(output: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Separate annotation lines (see ``SCRIPT_ANNOTATIONS``) from a script's output.

    Returns:
        The output without annotation lines, and the decoded annotations.
    """
    if ANNOTATION_MARKER not in output:
        return output, []
    kept: List[str] = []
    notes: List[Dict[str, Any]] = []
    for line in output.splitlines():
        marker_at = line.find(ANNOTATION_MARKER)
        if marker_at == -1:
            kept.append(line)
            continue
        try:
            note = json.loads(line[marker_at + len(ANNOTATION_MARKER):])
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if isinstance(note, dict):
            notes.append(note)
        if line[:marker_at].strip():
            kept.append(line[:marker_at])
    return "\n".join(kept), notes


def merge_annotations(data: Dict[str, Any], notes: Sequence[Dict[str, Any]]) -> None:
    """Merge annotation fields and warnings into a tool result, never over its own keys."""
    for note in notes:
        fields = note.get("fields")
        if isinstance(fields, dict):
            for key, value in fields.items():
                data.setdefault(key, value)
        extra = [w for w in note.get("warnings") or [] if isinstance(w, str)]
        if not extra:
            continue
        warnings = data.get("warnings")
        if warnings is None:
            data["warnings"] = list(extra)
        elif isinstance(warnings, list):
            warnings.extend(extra)
        else:
            data.setdefault("annotation_warnings", list(extra))


def _note_foreign_output(data: Dict[str, Any], other: str) -> None:
    """Record in ``data`` that stdout carried text besides the tool's JSON."""
    lines = len(other.splitlines())
    note = (
        f"Ignored {lines} line(s) of other output printed while the script ran "
        "(Kit's stdout is shared with every Python callback); see foreign_output."
    )
    warnings = data.get("warnings")
    if warnings is None:
        data["warnings"] = [note]
    elif isinstance(warnings, list):
        warnings.append(note)
    else:
        data["stdout_warning"] = note
    data["foreign_output"] = other[:FOREIGN_OUTPUT_REPORT_CHARS]


class IsaacScriptBase(LoggerMixin):
    """Owns the client, the sandbox policy, and script execution."""

    def __init__(
        self,
        client: IsaacSocketClient,
        settings: Optional[Settings] = None,
        client_resolver: Optional[Callable[[], IsaacSocketClient]] = None,
    ) -> None:
        """
        Initialize Isaac Sim tools.

        Args:
            client: Pre-configured IsaacSocketClient for TCP communication.
            settings: Configuration settings.
            client_resolver: Optional callable used to resolve the target client
                dynamically per request/session.
        """
        self._default_client = client
        self._client_resolver = client_resolver
        self.settings = settings or get_settings()
        # Enforced here rather than at the MCP boundary alone: the CLI calls
        # these methods directly, so a check that lives only in registration
        # governs one of the two entry points.
        self._path_policy = PathPolicy.from_settings(self.settings)

    @property
    def _client(self) -> IsaacSocketClient:
        """Resolve the target client for the current request."""
        if self._client_resolver is not None:
            return self._client_resolver()
        return self._default_client

    @_client.setter
    def _client(self, client: IsaacSocketClient) -> None:
        """Update the default client when using non-session-scoped routing."""
        self._default_client = client

    def _sandbox_denial(
        self, path: Optional[str], *, write: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Return an error payload when ``path`` is outside the sandbox.

        ``None`` means "carry on" — either the path is allowed, or there is no
        path to police (an optional save-as target that was not supplied). A
        caller that carries on must embed ``self._path_policy.authorize(path)``
        in the generated script, never the raw string: Kit resolves a relative
        path against its own working directory, not the project root the
        containment test used.

        Args:
            path: Path or URL supplied by the caller, or None.
            write: Whether the tool writes to the location.

        Returns:
            The SandboxError payload, or None when the path may be used.
        """
        return self._path_policy.denial(path, write=write)

    @staticmethod
    def _refusal(message: str, **details: Any) -> Dict[str, Any]:
        """Return the RefusedOperation payload for a known-fatal request.

        Args:
            message: What was refused and which flag, if any, overrides it.
            **details: Structured context for the caller (paths, keys, ids).

        Returns:
            An ErrorResponse dict with ``error_type`` ``RefusedOperation``.
        """
        return ErrorResponse(
            error=message,
            error_type=error_types.REFUSED_OPERATION,
            details=details or None,
        ).model_dump()

    @staticmethod
    def _script_failure(
        result: ScriptResult, raised: str = error_types.ENGINE_ERROR
    ) -> Dict[str, Any]:
        """Return the error envelope for a script that failed in or on the way to Isaac Sim.

        A failure of the transport (``error_types.ISAAC_TRANSPORT_FAILURES``, or
        one the client marks as its own) has its type. An exception the script
        raised inside Kit is ``raised`` whatever its name: a ``TypeError`` or
        an ``OSError`` there is the engine's failure (or the agent's script's),
        not the caller's arguments or a broken connection.

        Args:
            result: A ScriptResult whose ``success`` is False.
            raised: The type of an exception the script raised:
                ``EngineError`` for a tool's own script, ``ScriptError`` for
                an agent's.

        Returns:
            An ErrorResponse dict carrying the message, the traceback and the
            name the transport reported as ``details.exception_type``.
        """
        name = result.error_name or "UnknownError"
        error_type = error_types.engine_failure_type(
            name, transport_failure=result.transport_failure, raised=raised
        )
        details: Dict[str, Any] = {}
        if result.traceback:
            details["traceback"] = result.traceback
        if name != error_type:
            details["exception_type"] = name
        return ErrorResponse(
            error=result.error_value or "Script execution failed",
            error_type=error_type,
            details=details or None,
        ).model_dump()

    @staticmethod
    def _bridge_response_envelope(response: Dict[str, Any]) -> Dict[str, Any]:
        """Turn a bridge response frame into a tool envelope.

        An ``ok`` frame yields its payload, marked by the same rule as the
        script path: a typed action that reports ``timeline_error`` beside its
        other sections is a failure. Any other frame yields an ErrorResponse
        naming the remote error.

        Args:
            response: The decoded bridge response.

        Returns:
            The payload dict or an ErrorResponse dict.
        """
        if response.get("status") == "ok":
            result_payload = response.get("payload", {})
            if not isinstance(result_payload, dict):
                return ErrorResponse(
                    error="Bridge response payload must be an object.",
                    error_type=error_types.BRIDGE_PROTOCOL_ERROR,
                ).model_dump()
            return apply_success_from_error(result_payload)

        error = response.get("error", {})
        if not isinstance(error, dict):
            error = {}
        # The bridge's own refusals keep their meaning; the name of an
        # exception its handler raised is the engine's failure.
        name = str(error.get("name") or "BridgeError")
        error_type = error_types.engine_failure_type(name)
        details: Dict[str, Any] = {}
        if error.get("traceback"):
            details["traceback"] = error.get("traceback")
        if name != error_type:
            details["exception_type"] = name
        return ErrorResponse(
            error=str(error.get("message", "Bridge request failed")),
            error_type=error_type,
            details=details or None,
        ).model_dump()

    def _heavy_script_timeout(self) -> float:
        """Script budget for operations known to outlast the default.

        Opening a stage, a camera's first render (shader compilation) and
        enabling a large extension routinely take longer than the default
        29 s, and a timeout there reports failure for work Kit then finishes.
        A longer configured socket timeout is never shortened.
        """
        configured = getattr(self._client, "script_timeout_seconds", None)
        if isinstance(configured, bool) or not isinstance(configured, (int, float)):
            configured = 0.0
        return max(float(configured), HEAVY_SCRIPT_TIMEOUT_SECONDS)

    async def _execute_json_script(
        self,
        script: str,
        transport_mode: str = "default",
        *,
        timeout_seconds: Optional[float] = None,
        annotations: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """
        Execute a Python script in Isaac Sim and parse JSON from stdout.

        The script MUST print exactly one JSON object via json.dumps(), last.
        This helper handles connectivity errors, timeouts, parse failures,
        and script-level exceptions uniformly. Text other Python code printed
        into the same capture is set aside (see :func:`extract_tool_json`)
        and reported under ``foreign_output`` instead of failing the call.

        Args:
            script: Python source code that prints a JSON object to stdout.
            transport_mode: Transport routing -- "default" uses client's preferred
                transport, "bridge_only" forces the bridge, "vscode_only" forces
                the VS Code socket.
            timeout_seconds: Script budget for this call; the client's default
                when None.
            annotations: Names from ``SCRIPT_ANNOTATIONS`` whose checks run
                after the script and add fields and warnings to a successful
                result ("usd_readback" for readers of poses,
                "usd_velocity_readback" for readers of velocities too,
                "paused_edit" for edits).

        Returns:
            Parsed dict from stdout JSON, or an ErrorResponse dict.
        """
        if annotations:
            script = "\n\n".join(
                [script.rstrip("\n")] + [SCRIPT_ANNOTATIONS[name] for name in annotations]
            )
        # Passed only when set, so a client double with the plain
        # execute(code) signature keeps working for every other tool.
        kwargs: Dict[str, Any] = (
            {} if timeout_seconds is None else {"timeout_seconds": timeout_seconds}
        )
        try:
            if transport_mode == "bridge_only":
                result = await self._client.execute_bridge_script_only(script, **kwargs)
            elif transport_mode == "vscode_only":
                result = await self._client.execute_vscode_only(script, **kwargs)
            else:
                result = await self._client.execute(script, **kwargs)
        except TimeoutError as exc:
            # Before OSError, which TimeoutError subclasses.
            return ErrorResponse(
                error=str(exc)
                or f"Isaac Sim at {self._client.address} did not answer in time; the "
                "script may still be running.",
                **error_types.exception_fields(exc),
            ).model_dump()
        except (ConnectionError, OSError) as exc:
            # Refused, reset by the peer, or broken mid-reply: the shared class,
            # with the transport's own name as details.exception_type.
            return ErrorResponse(
                error=str(exc) or f"The connection to Isaac Sim at {self._client.address} failed.",
                **error_types.exception_fields(exc),
            ).model_dump()
        except Exception as exc:
            logger.error("Script execution failed: %s", exc, exc_info=True)
            return ErrorResponse(
                error=str(exc) or type(exc).__name__,
                **error_types.exception_fields(exc),
            ).model_dump()

        if not result.success:
            return self._script_failure(result)

        output, notes = split_annotations(result.output)
        output = output.strip()
        if not output:
            return ErrorResponse(
                error="Script produced no output",
                error_type=error_types.EMPTY_OUTPUT,
            ).model_dump()

        data, other_output = extract_tool_json(output)
        if not isinstance(data, dict):
            try:
                json.loads(output)
            except json.JSONDecodeError as exc:
                reason = str(exc)
            else:
                reason = f"expected a JSON object, got {type(data).__name__}"
            # The engine's output, not the caller's input, failed to parse.
            return ErrorResponse(
                error=f"Failed to parse script output as JSON: {reason}",
                error_type=error_types.ENGINE_ERROR,
                details={"raw_output": output[:2000], "exception_type": "JSONDecodeError"},
            ).model_dump()
        if other_output:
            _note_foreign_output(data, other_output)
        if notes and data.get("error") is None:
            merge_annotations(data, notes)
        if data.get("error") is not None and not data.get("error_type") and data.get("success") is not True:
            data["error_type"] = script_failure_type(data["error"])
        # A script that sets `success` itself is trusted; otherwise `error`
        # or any `*_error` key marks the call failed, so a partial result
        # (`syntheticdata_error` beside a valid listing) is not reported green.
        return apply_success_from_error(data)

    async def _execute_bridge_action(
        self,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        read_timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a typed bridge action without falling back to VS Code.

        Args:
            action: Bridge action name.
            payload: Action payload.
            read_timeout: How long to wait for the reply, for actions whose
                duration scales with their input; the bridge timeout when None.

        Returns:
            Parsed payload dict on success, an error dict on bridge-only
            failure, or None when the action is unavailable and the caller
            should try a bridge-script fallback.
        """
        if not self._client.bridge_enabled:
            return None

        # Passed only when set, so client doubles with the plain signature work.
        kwargs: Dict[str, Any] = {} if read_timeout is None else {"read_timeout": read_timeout}
        try:
            response = await self._client.bridge_request(action, payload or {}, **kwargs)
        except BridgeRequestDeliveredError as exc:
            # The bridge already holds the request and may have acted on it
            # (a typed create/delete/simulation_control is not idempotent).
            # Running the script fallback now would apply it a second time.
            return ErrorResponse(
                error=str(exc),
                **error_types.exception_fields(exc),
            ).model_dump()
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            # The bridge is unreachable, not the action unsupported. When a
            # script transport sits below us, defer to it exactly as an
            # UnknownAction does — returning an envelope here is final to the
            # caller and strands the tool on a closed port while ping, which
            # does fall back, still reports the instance reachable.
            if self._client.fallback_to_vscode:
                logger.debug(
                    "Bridge action %s unreachable (%s); deferring to script path",
                    action,
                    exc,
                )
                return None
            return ErrorResponse(
                error=str(exc),
                **error_types.exception_fields(exc),
            ).model_dump()
        except ValueError as exc:
            # Protocol-level failure (oversized request, malformed frame). The
            # script path would not do better, so surface it.
            return ErrorResponse(
                error=str(exc),
                **error_types.exception_fields(exc),
            ).model_dump()

        if (
            response.get("status") != "ok"
            and str(response.get("error", {}).get("name", "BridgeError")) == "UnknownAction"
        ):
            # An older bridge without this action; the caller falls back.
            return None
        return self._bridge_response_envelope(response)

    @property
    def _raw_script_transport_mode(self) -> str:
        """Choose the transport for raw script execution.

        The bridge is preferred whenever it is enabled: it is the only
        transport whose executor can bound a script with a timeout and stop
        it on request, so a runaway loop cannot take Kit down with it. The
        stock socket stays as the fallback ``execute`` already provides.
        """
        return "default"
