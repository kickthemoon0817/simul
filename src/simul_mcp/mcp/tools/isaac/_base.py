"""Transport and policy machinery every Isaac tool mixin builds on."""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ....adapters import IsaacSocketClient, ScriptResult
from ....config import Settings, get_settings
from ....logging import LoggerMixin
from ....utils.paths import PathPolicy
from ...schemas.common import ErrorResponse
from ._shared import (
    BULK_GEOMETRY_ATTRIBUTES,
    LOG_SCAN_WINDOW_BYTES,
    MAX_CAPTURE_DIMENSION,
    MAX_INLINE_CAPTURE_BYTES,
    MAX_RETAINED_CAPTURES,
    MAX_SCRIPT_BYTES,
    PRIM_DETAIL_ASPECTS,
    FloatList,
    _pyval,
    logger,
)


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
        if path is None or self._path_policy.is_allowed(path, write=write):
            return None
        return ErrorResponse(
            error="File path is not allowed by sandbox policy",
            error_type="SandboxError",
            details=self._path_policy.denial_details(path, write=write),
        ).model_dump()

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
            error_type="RefusedOperation",
            details=details or None,
        ).model_dump()

    async def _execute_json_script(
        self, script: str, transport_mode: str = "default"
    ) -> Dict[str, Any]:
        """
        Execute a Python script in Isaac Sim and parse JSON from stdout.

        The script MUST print exactly one JSON object via json.dumps().
        This helper handles connectivity errors, timeouts, parse failures,
        and script-level exceptions uniformly.

        Args:
            script: Python source code that prints a JSON object to stdout.
            transport_mode: Transport routing -- "default" uses client's preferred
                transport, "bridge_only" forces the bridge, "vscode_only" forces
                the VS Code socket.

        Returns:
            Parsed dict from stdout JSON, or an ErrorResponse dict.
        """
        try:
            if transport_mode == "bridge_only":
                result = await self._client.execute_bridge_script_only(script)
            elif transport_mode == "vscode_only":
                result = await self._client.execute_vscode_only(script)
            else:
                result = await self._client.execute(script)
        except ConnectionRefusedError as exc:
            return ErrorResponse(
                error=str(exc),
                error_type="ConnectionError",
            ).model_dump()
        except TimeoutError as exc:
            return ErrorResponse(
                error=str(exc),
                error_type="TimeoutError",
            ).model_dump()
        except Exception as exc:
            logger.error("Script execution failed: %s", exc, exc_info=True)
            return ErrorResponse(
                error=str(exc), error_type="Exception"
            ).model_dump()

        if not result.success:
            return ErrorResponse(
                error=result.error_value or "Script execution failed",
                error_type=result.error_name or "RuntimeError",
                details=(
                    {"traceback": result.traceback}
                    if result.traceback
                    else None
                ),
            ).model_dump()

        output = result.output.strip()
        if not output:
            return ErrorResponse(
                error="Script produced no output",
                error_type="EmptyOutput",
            ).model_dump()

        try:
            data: Dict[str, Any] = json.loads(output)
            # Wrapper-level success-flag normalization. If the script
            # already set `success` explicitly, honour it. Otherwise:
            # presence of `error` ⇒ failure (this closes the issue #35
            # shape across every Isaac tool whose script emits only an
            # error key — pre-fix, ~30+ such sites all returned the
            # misleading `{"error": "...", "success": true}` payload).
            # No `error` and no explicit `success` ⇒ success.
            if "success" not in data:
                data["success"] = "error" not in data
            return data
        except json.JSONDecodeError as exc:
            return ErrorResponse(
                error=f"Failed to parse script output as JSON: {exc}",
                error_type="JSONDecodeError",
                details={"raw_output": output[:2000]},
            ).model_dump()

    async def _execute_bridge_action(
        self, action: str, payload: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a typed bridge action without falling back to VS Code.

        Returns:
            Parsed payload dict on success, an error dict on bridge-only
            failure, or None when the action is unavailable and the caller
            should try a bridge-script fallback.
        """
        if not self._client.bridge_enabled:
            return None

        try:
            response = await self._client.bridge_request(action, payload or {})
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
                error_type=type(exc).__name__,
            ).model_dump()
        except ValueError as exc:
            # Protocol-level failure (oversized request, malformed frame). The
            # script path would not do better, so surface it.
            return ErrorResponse(
                error=str(exc),
                error_type=type(exc).__name__,
            ).model_dump()

        if response.get("status") == "ok":
            result_payload = response.get("payload", {})
            if not isinstance(result_payload, dict):
                return ErrorResponse(
                    error="Bridge response payload must be an object.",
                    error_type="BridgeProtocolError",
                ).model_dump()
            result_payload.setdefault("success", True)
            return result_payload

        error = response.get("error", {})
        error_name = str(error.get("name", "BridgeError"))
        if error_name == "UnknownAction":
            return None
        return ErrorResponse(
            error=str(error.get("message", "Bridge request failed")),
            error_type=error_name,
            details={"traceback": error.get("traceback")} if error.get("traceback") else None,
        ).model_dump()

    @property
    def _raw_script_transport_mode(self) -> str:
        """Route raw-script fallbacks away from the typed bridge by default."""
        if self._client.bridge_enabled:
            return "vscode_only"
        return "default"

