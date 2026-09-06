"""Cross-domain tools: aspect dispatch and raw script execution."""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ....adapters import IsaacSocketClient, ScriptResult
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
from .._meta import tool_meta


class CoreToolsMixin:
    @tool_meta(
        name="get_isaac_prim_detail",
        description=(
            "Read one or more aspects of a prim in a single call. Aspects: info, transform, "
            "ancestors, relationships, variants, bounding_box, mesh, light, material, "
            "rigid_body, collision, joint, mass, animation. Defaults to info."
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
            aspects: Which aspects to read. Defaults to ``["info"]``. Valid
                     names are the keys of PRIM_DETAIL_ASPECTS.

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
                error_type="ValueError",
            ).model_dump()

        # One bridge round trip reads every aspect; the per-aspect loop below
        # is the fallback for a bridge that is off, unreachable, or too old to
        # know the action, and costs one connection per aspect.
        bridge_result = await self._execute_bridge_action(
            "get_prim_detail", {"prim_path": prim_path, "aspects": requested}
        )
        if bridge_result is not None:
            return bridge_result

        detail: Dict[str, Any] = {
            "success": True,
            "prim_path": prim_path,
            "aspects": requested,
        }
        for aspect in requested:
            method = getattr(self, PRIM_DETAIL_ASPECTS[aspect])
            # Aspects are independent reads: one that fails is reported in
            # place rather than discarding the rest of the answer.
            detail[aspect] = await method(prim_path)
        return detail

    @tool_meta(
        name="execute_isaac_script",
        description=(
            "Execute arbitrary Python code inside a running Isaac Sim application. The code "
            "runs in Kit's Python scope with full access to omni.*, pxr.*, and isaacsim.* "
            "APIs. stdout is captured and returned. For structured results, print JSON via "
            "json.dumps(). Each execution is independent — always import modules at the top "
            "of every script. Call ping_isaac first to verify connectivity before sending "
            "scripts. Read the 'simul://isaac-sim/skills' resource for scripting patterns, "
            "API quick reference, and Isaac Sim 5.1 / 6.0 namespace migration notes."
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
                error_type="ScriptExecutionDisabled",
            ).model_dump()
        if len(code) > MAX_SCRIPT_BYTES:
            return ErrorResponse(
                error=(
                    f"Code payload too large ({len(code)} bytes, "
                    f"max {MAX_SCRIPT_BYTES})."
                ),
                error_type="PayloadTooLarge",
            ).model_dump()

        client = self._client
        try:
            if self._raw_script_transport_mode == "vscode_only":
                result = await client.execute_vscode_only(code)
            else:
                result = await client.execute(code)
        except ConnectionRefusedError:
            return ErrorResponse(
                error=(
                    f"Isaac Sim is not reachable at {client.address}. "
                    "Ensure Isaac Sim is running with the "
                    "isaacsim.code_editor.vscode extension enabled. "
                    "Use ping_isaac to verify connectivity."
                ),
                error_type="ConnectionError",
            ).model_dump()
        except TimeoutError:
            return ErrorResponse(
                error=(
                    f"Script execution timed out after "
                    f"{client.timeout_seconds}s on {client.address}. "
                    "The script may be too slow or Isaac Sim may be "
                    "unresponsive. Use ping_isaac to check if Isaac Sim is "
                    "still reachable."
                ),
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
                    {"traceback": result.traceback} if result.traceback else None
                ),
            ).model_dump()

        output = result.output.strip()
        if output:
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                # A generated script reports failure as {"error": ...}; stamping
                # success onto that contradicts it, and the caller reading
                # "success" is usually an LLM that will believe the flag.
                parsed.setdefault("success", "error" not in parsed)
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

