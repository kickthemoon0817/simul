"""Cross-domain tools: aspect dispatch and raw script execution."""

import json
from typing import Any, Dict, List, Optional

from ... import error_types
from ...registration._helpers import apply_success_from_error, utf8_byte_length
from ...schemas.common import ErrorResponse
from ._shared import (
    MAX_SCRIPT_BYTES,
    PRIM_DETAIL_ASPECTS,
    logger,
)
from .._meta import tool_meta


def _typed_aspect_failures(payload: Dict[str, Any], aspects: List[str]) -> Dict[str, Any]:
    """Give each failed aspect of a prim-detail answer a type from the vocabulary.

    An aspect is read on its own and fails in place, so its failure is a
    payload inside the answer that the envelope does not see. The bridge names
    it after the exception its reader raised inside Kit (an ``AttributeError``,
    a ``TypeError``): the engine's failure, kept as ``exception_type``.
    """
    for aspect in aspects:
        section = payload.get(aspect)
        if not isinstance(section, dict) or section.get("success") is not False:
            continue
        name = section.get("error_type")
        if isinstance(name, str) and name and not error_types.is_known(name):
            section["error_type"] = error_types.engine_failure_type(name)
            details = section.get("details")
            details = dict(details) if isinstance(details, dict) else {}
            details.setdefault("exception_type", name)
            section["details"] = details
        error_types.normalize_error_payload(section)
    return payload


class CoreToolsMixin:
    @tool_meta(
        name="get_isaac_prim_detail",
        description=(
            "Read one or more aspects of a prim in a single call. Aspects: info, transform, "
            "ancestors, relationships, variants, bounding_box, mesh, light, material, "
            "rigid_body, collision, joint, mass, animation. Defaults to info. transform "
            "(and info's transform) reports the world pose with 'local' and 'world' "
            "blocks: translation, rotation_quat [w, x, y, z], rotation_euler_xyz (degrees) "
            "and scale, factored from the composed matrix."
        ),
        read_only=True,
        idempotent=True,
    )
    async def get_isaac_prim_detail(
        self,
        prim_path: str,
        aspects: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Read one or more aspects of a prim in a single call.

        Args:
            prim_path: USD path of the prim to inspect.
            aspects: Aspect names to read (listed in the tool description).
                     Defaults to ``["info"]``.

        Returns:
            Dict with one entry per requested aspect, keyed by aspect name.
        """
        requested = list(aspects) if aspects else ["info"]
        unknown = [a for a in requested if a not in PRIM_DETAIL_ASPECTS]
        if unknown:
            return ErrorResponse(
                error=(
                    f"Unknown aspect(s): {', '.join(unknown)}. "
                    f"Valid aspects: {', '.join(sorted(PRIM_DETAIL_ASPECTS))}."
                ),
                error_type="ValidationError",
            ).model_dump()

        # One bridge round trip reads every aspect; the per-aspect loop below
        # is the fallback for a bridge that is off, unreachable, or too old to
        # know the action, and costs one connection per aspect.
        bridge_result = await self._execute_bridge_action(
            "get_prim_detail", {"prim_path": prim_path, "aspects": requested}
        )
        if bridge_result is not None:
            return _typed_aspect_failures(bridge_result, requested)

        detail: Dict[str, Any] = {
            "success": True,
            "prim_path": prim_path,
            "aspects": requested,
        }
        missing = f"Prim not found: {prim_path}"
        for aspect in requested:
            method = getattr(self, PRIM_DETAIL_ASPECTS[aspect])
            # Aspects are independent reads: one that fails is reported in
            # place rather than discarding the rest of the answer.
            detail[aspect] = await method(prim_path)
            if detail[aspect].get("error") == missing:
                # A missing prim fails every aspect alike. Answer the way the
                # bridge's get_prim_detail does (its PrimNotFound, which the
                # envelope maps onto NotFoundError), not with success: true
                # and the same error in each aspect.
                return ErrorResponse(error=missing, error_type="PrimNotFound").model_dump()
        return _typed_aspect_failures(detail, requested)

    @tool_meta(
        name="execute_isaac_script",
        description=(
            "Execute arbitrary Python code inside a running Isaac Sim application. The code "
            "runs in Kit's Python scope with full access to omni.*, pxr.*, and isaacsim.* "
            "APIs. stdout is captured and returned. For structured results, print JSON via "
            "json.dumps(). Executions are not isolated: every transport runs scripts in one "
            "shared namespace, so globals a script defines persist into later calls, "
            "including other agents' calls on the same instance. Import what you use in "
            "every script, keep helpers in functions, and don't depend on leftover globals. "
            "On Isaac Sim 5.1 with the bridge down, scripts over 32 KiB are refused. Call "
            "ping_isaac first to verify connectivity before sending scripts. Read the "
            "'simul://isaac-sim/skills' resource for scripting patterns, API quick "
            "reference, and Isaac Sim 5.1 / 6.0 namespace migration notes."
        ),
        read_only=False,
        destructive=True,
        script=True,
        hidden_parameters=("keep_raw_output",),
    )
    async def execute_script(
        self, code: str, keep_raw_output: bool = False
    ) -> Dict[str, Any]:
        """
        Execute raw Python inside the running Isaac Sim process.

        The single implementation of the raw-script path. Callers get the
        transport policy, the JSON unwrap and the error envelopes from here;
        rate limiting, instance locking, usage tracking and the session
        heartbeat come from the server wrapper that invokes it.

        Args:
            code: Python source to run in Isaac Sim's interpreter.
            keep_raw_output: Also return the script's stdout verbatim even when
                it parsed as a JSON object. Off by default because it doubles
                the payload; the CLI's --raw needs it, MCP callers do not.

        Returns:
            The script's JSON object when it printed one, otherwise its text
            output, or an error envelope.
        """
        if not self.settings.security.allow_script_execution:
            return ErrorResponse(
                error=(
                    "Arbitrary script execution is disabled by "
                    "security.allow_script_execution. Use the granular tools, or "
                    "set SECURITY__ALLOW_SCRIPT_EXECUTION=true to re-enable it."
                ),
                error_type=error_types.SCRIPT_EXECUTION_DISABLED,
            ).model_dump()
        code_bytes = utf8_byte_length(code)
        if code_bytes > MAX_SCRIPT_BYTES:
            return ErrorResponse(
                error=(
                    f"Code payload too large ({code_bytes} bytes in UTF-8, "
                    f"max {MAX_SCRIPT_BYTES})."
                ),
                error_type=error_types.PAYLOAD_TOO_LARGE,
            ).model_dump()

        client = self._client
        try:
            if self._raw_script_transport_mode == "vscode_only":
                result = await client.execute_vscode_only(code)
            else:
                result = await client.execute(code)
        except ConnectionRefusedError as exc:
            return ErrorResponse(
                error=(
                    f"Isaac Sim is not reachable at {client.address}. "
                    "Ensure Isaac Sim is running with the "
                    "isaacsim.code_editor.vscode extension enabled. "
                    "Use ping_isaac to verify connectivity."
                ),
                **error_types.exception_fields(exc),
            ).model_dump()
        except TimeoutError as exc:
            return ErrorResponse(
                error=(
                    f"Script execution timed out after "
                    f"{client.timeout_seconds}s on {client.address}. "
                    "The script may be too slow or Isaac Sim may be "
                    "unresponsive. Use ping_isaac to check if Isaac Sim is "
                    "still reachable."
                ),
                **error_types.exception_fields(exc),
            ).model_dump()
        except Exception as exc:
            # A reset mid-reply is a ConnectionError, a garbled reply an
            # EngineError: the shared classes, never the bare "Exception".
            logger.error("Script execution failed: %s", exc, exc_info=True)
            return ErrorResponse(
                error=str(exc) or type(exc).__name__,
                **error_types.exception_fields(exc),
            ).model_dump()

        if not result.success:
            # Whatever the agent's script raised (a NameError, an OSError, a
            # ConnectionRefusedError of its own) is a ScriptError, as on
            # Unreal, naming it; only the transport's own failures keep theirs.
            return self._script_failure(result, raised=error_types.SCRIPT_ERROR)

        output = result.output.strip()
        if output:
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                # A generated script reports failure as {"error": ...} or a
                # partial one as {"<section>_error": ...}; stamping success onto
                # that contradicts it, and the caller reading "success" is
                # usually an LLM that will believe the flag. Same rule as the
                # granular tools' script path.
                apply_success_from_error(parsed)
                if keep_raw_output:
                    parsed.setdefault("output", result.output)
                return parsed
            if parsed is not None:
                # A bare JSON value is not a payload, so keep the text too.
                return {
                    "success": True,
                    "output": result.output,
                    "parsed": parsed,
                }

        return {"success": True, "output": result.output}
