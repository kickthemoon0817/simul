"""
Simul 3D MCP Server implementation.

This module provides the main MCP server class with tool registry,
connection management, and 3D simulation/DCC integration based on FastMCP.
"""

import asyncio
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from fastmcp import FastMCP
from fastmcp.server.context import _current_context
from fastmcp.server.tasks import TaskConfig
from fastmcp.tools.tool import ToolResult
from mcp.types import ImageContent, TextContent, ToolAnnotations
from pydantic import BaseModel

from .. import __version__ as _source_version
from ..adapters import (
    BlenderRuntimeAdapter,
    HeadlessUSDAdapter,
    IsaacSocketClient,
    UnrealRuntimeAdapter,
    is_blender_available,
    is_headless_available,
    is_unreal_available,
)
from ..config import Settings, get_settings
from ..logging import LoggerMixin, get_logger
from ..resources import find_checkout_root, resource
from ..utils.paths import PathPolicy, SandboxDenied
from ..utils.timing import RateLimiter
from .registration._helpers import apply_success_from_error
from .result_budget import apply_result_budget
from .schemas.common import ErrorResponse
from .session_manager import CLAIM_TTL_SECONDS, SessionManager
from ..utils.discovery import DiscoveryDir
from .tools.isaac_tools import IsaacTools
from .usage_tracker import ToolUsageTracker

logger = get_logger(__name__)

try:
    from importlib.metadata import version as _pkg_version

    _PACKAGE_VERSION: str = _pkg_version("simul-mcp")
except Exception:
    _PACKAGE_VERSION = _source_version

_MCP_INSTRUCTIONS: str = (
    "Simul MCP provides tools for interacting with 3D simulation "
    "and DCC applications. Use Isaac Sim tools to control a running NVIDIA Isaac Sim "
    "instance — granular tools for scene inspection, prim "
    "manipulation, physics, simulation control, materials, "
    "viewport/camera, rendering, and asset/stage operations. "
    "Use USD tools to load, inspect, and edit Universal Scene "
    "Description files. "
    "Use Blender tools when a Blender runtime is connected. "
    "Use Unreal tools when an Unreal Engine instance is connected.\n\n"
    "TOOL SELECTION — prefer a granular tool when one covers the "
    "operation; they are cheaper and return structured results. For GUI/app "
    "state (windows, focus, selection, timeline) use get_isaac_ui_state; to "
    "inspect one window's widget tree use get_isaac_ui_window. Reach for "
    "execute_isaac_script when none does (custom extensions, replicator "
    "workflows, robotics APIs, warp kernels). Read the "
    "'simul://isaac-sim/skills' resource for scripting patterns and API "
    "reference when writing scripts.\n\n"
    "CONVENTIONS — Isaac Sim stages are Z-up with metersPerUnit=1, so "
    "positions and distances are metres, gravity points along -Z, and "
    "rotations are XYZ Euler degrees; get_isaac_stage_info reports the "
    "up_axis and meters_per_unit of a loaded file when they differ. Every "
    "tool parameter's units and conventions are in its schema description.\n\n"
    "ROUTING — the backend name appears somewhere inside the tool name, "
    "usually as an infix (create_isaac_prim, capture_unreal_viewport, "
    "get_blender_info):\n"
    "  Tools containing 'isaac' → require a running Isaac Sim instance (TCP socket).\n"
    "  Tools containing 'unreal' → require a connected Unreal Engine instance.\n"
    "  Tools containing 'blender' → require a connected Blender runtime.\n"
    "  Tools containing 'simready' (SimReady asset helpers) exist for both "
    "Blender and Unreal; each one's description names the runtime it targets.\n"
    "  Tools containing none of those names (load_usd_file, get_prim_info, "
    "create_prim, summarize_scene, etc.) → operate on local USD files via the "
    "headless adapter; they do NOT connect to any engine. The two "
    "*_tool_usage_stats tools are server metadata, not a backend.\n\n"
    "MULTI-INSTANCE — when multiple Isaac Sim applications are running:\n"
    "  1. Call list_isaac_instances to discover all running instances "
    "and see which stage each has loaded.\n"
    "  2. Call set_active_isaac_instance to switch within the current MCP session.\n"
    "  3. All subsequent Isaac tool calls in that same session route to that instance.\n"
    "  4. For containerized Isaac Sim, use the host-published bridge / VS Code ports, "
    "not the container-internal ports."
)

_FASTMCP_SUPPORTS_INSTRUCTIONS: bool = (
    "instructions" in inspect.signature(FastMCP).parameters
)

# Tool key of the bucket that caps one agent's calls across every tool.
_GLOBAL_RATE_BUCKET: str = "*"


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


class SimulMCPServer(LoggerMixin):
    """
    Simul 3D MCP Server.

    Unified MCP server for 3D simulation and DCC tools. Registers tools for
    Isaac Sim (live TCP), headless USD, Blender, and Unreal Engine backends.
    Supports multi-instance Isaac Sim discovery and active instance routing.
    """

    # All recognised backend names for --backends validation.
    ALL_BACKENDS: Set[str] = {"isaac", "unreal", "usd", "blender"}

    def __init__(
        self,
        settings: Optional[Settings] = None,
        backends: Optional[Set[str]] = None,
    ):
        """
        Initialize Simul 3D MCP Server.

        Args:
            settings: Configuration settings
            backends: Set of backend names to register MCP tools for.
                      ``None`` (default) registers all available backends.
                      Valid names: ``isaac``, ``unreal``, ``usd``, ``blender``.
        """
        self.settings = settings or get_settings()
        self._backends = backends  # None means "all available"
        self._path_policy = PathPolicy.from_settings(
            self.settings, project_root=find_checkout_root()
        )
        self._allowed_paths = self._resolve_allowed_paths()

        self.usage_tracker = ToolUsageTracker()
        self.session_manager = SessionManager()
        self._session_routes: Dict[str, IsaacSessionRoute] = {}
        self._read_only_tools: Dict[str, bool] = {}
        self._isaac_instance_locks: Dict[str, asyncio.Lock] = {}
        # Token buckets keyed by (agent, tool); the per-agent ceiling across
        # every tool uses _GLOBAL_RATE_BUCKET as the tool key.
        self._rate_limiters: Dict[Tuple[str, str], RateLimiter] = {}
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
        self._tool_timeout = self.settings.server.timeout
        # How long a call waits for an instance already in use. Generous enough
        # for ordinary contention, short enough that a caller is not left
        # guessing through a 1000-frame step.
        self._instance_lock_timeout = float(self.settings.server.timeout)

        # Initialize adapters
        self.headless_adapter = (
            HeadlessUSDAdapter(self.settings) if is_headless_available() else None
        )

        # Multi-instance Isaac Sim registry
        self._isaac_clients: Dict[str, IsaacSocketClient] = {}
        self._active_instance: str = "default"
        default_client = self._build_isaac_client(
            socket_host=self.settings.isaac_sim.socket_host,
            socket_port=self.settings.isaac_sim.socket_port,
            socket_timeout=self.settings.isaac_sim.socket_timeout,
            bridge_enabled=self.settings.isaac_sim.bridge_enabled,
            bridge_host=self.settings.isaac_sim.bridge_host,
            bridge_port=self.settings.isaac_sim.bridge_port,
            bridge_timeout=self.settings.isaac_sim.bridge_timeout,
            bridge_fallback_to_vscode=self.settings.isaac_sim.bridge_fallback_to_vscode,
        )
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

        self.blender_adapter = (
            BlenderRuntimeAdapter(self.settings) if is_blender_available() else None
        )
        self.unreal_adapter = (
            UnrealRuntimeAdapter(self.settings)
            if UnrealRuntimeAdapter is not None
            else None
        )

        # Initialize FastMCP server
        mcp_kwargs: Dict[str, Any] = {
            "name": "Simul – 3D Simulation & DCC Tools",
            "version": _PACKAGE_VERSION,
        }
        if _FASTMCP_SUPPORTS_INSTRUCTIONS:
            mcp_kwargs["instructions"] = _MCP_INSTRUCTIONS

        self.mcp = FastMCP(
            **mcp_kwargs,
        )

        # Tag every CallTool request with a fresh correlation id and emit an
        # audit row. Done as FastMCP middleware (vs. wrapping each tool) so it
        # composes with all decorator overloads, sync + async tool bodies, and
        # any future tools without per-registration changes.
        from ..logging import build_request_context_middleware

        self.mcp.add_middleware(build_request_context_middleware())

        # Register tools and resources
        self._register_tools()
        self._register_resources()

        self.logger.info("Simul 3D MCP Server initialized")

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
            if tool_name == _GLOBAL_RATE_BUCKET:
                limiter = RateLimiter(
                    self._global_rate_limit_rate, self._global_rate_limit_burst
                )
            else:
                limiter = RateLimiter(self._rate_limit_rate, self._rate_limit_burst)
            self._rate_limiters[key] = limiter
        return limiter

    def _check_rate_limit(
        self, tool_name: str, agent_id: Optional[str] = None
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

        Returns:
            The RateLimitError payload carrying ``retry_after_seconds``, or
            None when the call may proceed.
        """
        if not self._rate_limit_enabled:
            return None
        agent = agent_id or self._resolve_agent_id(None)
        tool_bucket = self._get_rate_limiter(tool_name, agent)
        agent_bucket = self._get_rate_limiter(_GLOBAL_RATE_BUCKET, agent)
        tool_wait = tool_bucket.seconds_until_available()
        agent_wait = agent_bucket.seconds_until_available()
        if tool_wait <= 0.0 and agent_wait <= 0.0:
            tool_bucket.acquire()
            agent_bucket.acquire()
            return None

        per_tool = tool_wait >= agent_wait
        retry_after = max(round(max(tool_wait, agent_wait), 3), 0.001)
        security = self.settings.security
        payload = ErrorResponse(
            error=f"Rate limit exceeded; retry in {retry_after} s",
            error_type="RateLimitError",
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
        """Return the current FastMCP session id when available."""
        ctx = self._get_request_context()
        if ctx is None:
            return None
        try:
            return ctx.session_id
        except Exception:
            return None

    def _get_or_create_session_route(self, session_id: str) -> IsaacSessionRoute:
        """Return the per-session Isaac routing state."""
        route = self._session_routes.get(session_id)
        if route is None:
            route = IsaacSessionRoute(
                session_id=session_id,
                active_instance=self._active_instance,
            )
            self._session_routes[session_id] = route
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
        """Resolve a stable per-session agent id."""
        if agent_id:
            return agent_id
        session_id = self._get_request_session_id()
        if session_id:
            return session_id
        return f"agent-{id(self):x}"

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

    def _foreign_claim(
        self, instance_name: str, caller_agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the live claim another agent holds on ``instance_name``, when enforced.

        Args:
            instance_name: Registry name of the Isaac instance being used.
            caller_agent_id: The agent making the request.

        Returns:
            The holder's session record when ``isaac_sim.enforce_claims`` is on
            and a live claim by a different agent exists, otherwise ``None``.
            Expired claims are pruned by the session manager, so a stale holder
            never blocks.
        """
        if not self.settings.isaac_sim.enforce_claims:
            return None
        client = self._isaac_clients.get(instance_name)
        if client is None:
            return None
        sessions = self.session_manager.get_instance_session(client._port).get_status()["sessions"]
        for session in sessions:
            if session.get("agent_id") != caller_agent_id:
                return session
        return None

    def _claimed_error(self, instance_name: str, holder: Dict[str, Any]) -> Dict[str, Any]:
        """Build the InstanceClaimed payload naming the holder and the way forward."""
        expires_in = max(0.0, CLAIM_TTL_SECONDS - (time.time() - float(holder.get("last_active", 0.0))))
        return ErrorResponse(
            error=(
                f"Isaac instance {instance_name!r} is claimed by agent "
                f"{holder.get('agent_id')!r} for {holder.get('purpose', '')!r}; "
                "isaac_sim.enforce_claims refuses mutating tools from other agents while "
                "that claim is live."
            ),
            error_type="InstanceClaimed",
            details={
                "instance": instance_name,
                "holder_agent_id": holder.get("agent_id"),
                "holder_purpose": holder.get("purpose", ""),
                "claim_expires_in_seconds": round(expires_in, 1),
                "hint": (
                    "Read-only tools still work. Ask the holder to call "
                    "release_isaac_instance, or wait for the claim to expire "
                    f"({int(CLAIM_TTL_SECONDS)} s without activity), then call "
                    "claim_isaac_instance yourself. list_isaac_instances shows other instances."
                ),
            },
        ).model_dump()

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
                read_only = bool(getattr(annotations, "readOnlyHint", False))
        self._read_only_tools[tool_name] = read_only
        return read_only

    async def _exec_isaac(
        self,
        tool_name: str,
        coro: Coroutine[Any, Any, Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
        script_sha256: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute an Isaac Sim tool coroutine with rate limiting,
        unified error handling, and usage tracking.

        Args:
            tool_name: Name of the tool for rate limiting.
            coro: Awaitable coroutine returned by an IsaacTools method.
            params: Optional dict of call parameters for usage logging.
            script_sha256: Digest of the agent-authored source for script tools.

        Returns:
            Tool result dict or error response dict.
        """
        agent_id = self._resolve_agent_id(None)
        rate_error = self._check_rate_limit(tool_name, agent_id)
        if rate_error is not None:
            self.usage_tracker.record(
                tool_name,
                0.0,
                False,
                params=params,
                error="rate_limited",
                agent_id=agent_id,
                script_sha256=script_sha256,
            )
            # Caller already built the coroutine; nothing will await it now.
            coro.close()
            return self._as_text_result(rate_error)
        instance_name = self._get_effective_instance_name()
        # --- claim enforcement (isaac_sim.enforce_claims) ---
        if self.settings.isaac_sim.enforce_claims and not await self._tool_is_read_only(tool_name):
            holder = self._foreign_claim(instance_name, self._request_agent_id())
            if holder is not None:
                self.usage_tracker.record(
                    tool_name,
                    0.0,
                    False,
                    params=params,
                    error="instance_claimed",
                    agent_id=agent_id,
                    script_sha256=script_sha256,
                )
                coro.close()
                return self._as_text_result(self._claimed_error(instance_name, holder))
        # --- end claim enforcement ---
        lock = self._get_instance_lock(instance_name)
        try:
            # Bounded, so a caller queued behind a long step learns the instance
            # is busy instead of waiting until its own client gives up. Without
            # this the only signal is a timeout, which reads as "unreachable".
            await asyncio.wait_for(lock.acquire(), timeout=self._instance_lock_timeout)
        except asyncio.TimeoutError:
            self.usage_tracker.record(
                tool_name,
                0.0,
                False,
                params=params,
                error="instance_busy",
                agent_id=agent_id,
                script_sha256=script_sha256,
            )
            coro.close()
            return self._as_text_result(
                ErrorResponse(
                    error=(
                        f"Isaac instance {instance_name!r} is busy with another "
                        "call. It is running, not unreachable — retry shortly."
                    ),
                    error_type="InstanceBusy",
                ).model_dump()
            )
        try:
            t0 = time.monotonic()
            try:
                result = await coro
                duration_ms = (time.monotonic() - t0) * 1000
                success = not result.get("error")
                self.usage_tracker.record(
                    tool_name,
                    duration_ms,
                    success,
                    params=params,
                    error=result.get("error") if not success else None,
                    agent_id=agent_id,
                    script_sha256=script_sha256,
                )
                binding = self._get_active_binding()
                if binding is not None:
                    self.session_manager.get_instance_session(binding.port).heartbeat(
                        binding.agent_id, tool_name
                    )
                    binding.last_heartbeat = time.time()
                return self._as_text_result(result)
            except Exception as exc:
                duration_ms = (time.monotonic() - t0) * 1000
                self.usage_tracker.record(
                    tool_name,
                    duration_ms,
                    False,
                    params=params,
                    error=str(exc),
                    agent_id=agent_id,
                    script_sha256=script_sha256,
                )
                logger.error("Isaac tool %s failed: %s", tool_name, exc)
                return self._as_text_result(
                    ErrorResponse(
                        error=str(exc), error_type=type(exc).__name__
                    ).model_dump()
                )
        finally:
            lock.release()

    def _resolve_allowed_paths(self) -> List[Path]:
        return self._path_policy.allowed_roots

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
        if path_str is None or self._path_policy.is_allowed(path_str, write=write):
            return None
        return ErrorResponse(
            error="File path is not allowed by sandbox policy",
            error_type="SandboxError",
            details=self._path_policy.denial_details(path_str, write=write),
        ).model_dump()

    def _validate_input(
        self, model: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, Dict[str, Any]]:
        try:
            return model(**kwargs)
        except Exception as e:
            return ErrorResponse(
                error=str(e), error_type="ValidationError", details={"input": kwargs}
            ).model_dump()

    def _validate_output(
        self,
        result: Any,
        models: Tuple[Type[BaseModel], ...],
        tool_name: str,
    ) -> Dict[str, Any]:
        if isinstance(result, BaseModel):
            payload: Any = result.model_dump()
        else:
            payload = result

        if (
            isinstance(payload, dict)
            and payload.get("success") is False
            and payload.get("error")
        ):
            return payload

        if not isinstance(payload, dict):
            return ErrorResponse(
                error="Tool returned invalid response type",
                error_type="ValidationError",
                details={"type": type(payload).__name__},
            ).model_dump()

        for model in models:
            try:
                model(**payload)
                return payload
            except Exception:
                continue

        return ErrorResponse(
            error="Tool response failed schema validation",
            error_type="ValidationError",
            details={"tool": tool_name},
        ).model_dump()

    async def _exec_backend(
        self,
        tool_name: str,
        adapter: Any,
        adapter_label: str,
        response_model: Type[BaseModel],
        call: Callable[[Any], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]],
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Run a USD, Blender or Unreal tool with the shared envelope.

        This is the one place every non-Isaac tool passes through, so it
        carries everything a tool owes the operator and the caller: the
        per-agent rate limit, the usage record, the availability check, the
        session scope, success normalisation, schema validation, sandbox and
        exception wrapping, the result budget and single transmission. Tools
        that hand-rolled this block drifted on every one of those points.

        Args:
            tool_name: Registered tool name, used for rate limiting and logs.
            adapter: Runtime adapter, or None when the backend is absent.
            adapter_label: Human-readable backend name for the error message.
            response_model: Schema a successful payload is validated against.
            call: Receives an open session and returns the payload, or an
                error envelope (``success`` False with ``error``) that is
                passed through as is. May be sync or async.
            params: Call parameters worth keeping in the usage log.

        Returns:
            The payload as one JSON content block, plus an image block when
            it carries one.
        """
        agent_id = self._resolve_agent_id(None)
        rate_error = self._check_rate_limit(tool_name, agent_id)
        if rate_error is not None:
            self.usage_tracker.record(
                tool_name, 0.0, False, params=params, error="rate_limited", agent_id=agent_id
            )
            return self._as_text_result(rate_error)

        started = time.monotonic()
        payload = await self._run_backend_call(
            tool_name, adapter, adapter_label, response_model, call
        )
        duration_ms = (time.monotonic() - started) * 1000
        error = payload.get("error")
        success = not error and payload.get("success", True) is not False
        self.usage_tracker.record(
            tool_name,
            duration_ms,
            success,
            params=params,
            error=str(error) if error else None,
            agent_id=agent_id,
        )
        return self._as_text_result(payload)

    async def _run_backend_call(
        self,
        tool_name: str,
        adapter: Any,
        adapter_label: str,
        response_model: Type[BaseModel],
        call: Callable[[Any], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]],
    ) -> Dict[str, Any]:
        """Open a session, run ``call`` and validate what it returns.

        Args:
            tool_name: Registered tool name for logs and validation errors.
            adapter: Runtime adapter, or None when the backend is absent.
            adapter_label: Human-readable backend name for the error message.
            response_model: Schema a successful payload is validated against.
            call: Receives the open session and returns the payload.

        Returns:
            The validated payload, or an error envelope.
        """
        models = (response_model, ErrorResponse)
        try:
            if adapter is None or not adapter.is_available():
                return self._validate_output(
                    ErrorResponse(
                        error=f"{adapter_label} runtime not available",
                        error_type="RuntimeError",
                    ).model_dump(),
                    models,
                    tool_name,
                )

            with adapter.create_session() as session:
                # Unreal sessions are async, Blender sessions are sync; accept
                # both so one envelope serves every backend.
                payload = call(session)
                if inspect.isawaitable(payload):
                    payload = await payload
                apply_success_from_error(payload)
                if payload.get("success") is False and payload.get("error"):
                    # An error envelope built by the tool (input validation,
                    # not-found, script failure) keeps its own fields; forcing
                    # it through the success schema would only replace the
                    # message with a pydantic complaint.
                    return self._validate_output(payload, models, tool_name)
                return self._validate_output(
                    response_model(**payload).model_dump(), models, tool_name
                )
        except SandboxDenied as exc:
            return self._validate_output(
                ErrorResponse(
                    error="File path is not allowed by sandbox policy",
                    error_type="SandboxError",
                    details=exc.details,
                ).model_dump(),
                models,
                tool_name,
            )
        except Exception as exc:
            self.logger.error("Error in %s: %s", tool_name, exc)
            return self._validate_output(
                ErrorResponse(error=str(exc), error_type="Exception").model_dump(),
                models,
                tool_name,
            )

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

        Args:
            payload: The tool's result dict.

        Returns:
            One JSON text block, preceded by an image block when present.
        """
        content: List[Any] = []
        image = payload.get("image_base64")
        if isinstance(image, str) and image:
            payload = {
                key: value
                for key, value in payload.items()
                if key not in ("image_base64", "encoding")
            }
            payload["image_attached"] = True
            image_format = str(payload.get("format", "png")).lower()
            mime_type = (
                "image/jpeg" if image_format in ("jpg", "jpeg") else f"image/{image_format}"
            )
            content.append(ImageContent(type="image", data=image, mimeType=mime_type))
        text = json.dumps(apply_result_budget(payload), default=str)
        content.append(TextContent(type="text", text=text))
        return ToolResult(content=content)

    def _script_tool(self, **tool_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register an arbitrary-code tool, or leave it off the surface.

        ``security.allow_script_execution`` is the operator's switch for the
        agent-authored code path. Granular tools are generated scripts and stay
        registered either way; only the three ``execute_*_script`` tools go
        through here.

        Args:
            **tool_kwargs: Keyword arguments for ``FastMCP.tool``.

        Returns:
            The FastMCP registration decorator, or an identity decorator when
            script execution is disabled.
        """
        if self.settings.security.allow_script_execution:
            return self.mcp.tool(**tool_kwargs)

        def unregistered(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return unregistered

    def _tool_annotations(
        self,
        read_only: bool,
        idempotent: bool,
        open_world: bool,
        destructive: bool = False,
    ) -> Optional[Any]:
        """Build the MCP annotations for one tool.

        Every hint the call site computes is preserved: a hint is omitted only
        when it equals the value a client assumes for a missing hint
        (destructive and idempotent false, open world true), so nothing is
        dropped and the listing stays small. readOnlyHint is always emitted
        because auto-approval policies key on it.

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
        if destructive:
            annotations["destructiveHint"] = True
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

        FastMCP 3.x can import TaskConfig without the full 'tasks' extra
        installed, but raises ImportError at tool-registration time when
        the ``docket`` runtime (``fastmcp[tasks]``) is absent.  Check
        for actual runtime availability before returning a config.
        """
        if TaskConfig:
            try:
                from fastmcp.server.dependencies import is_docket_available

                if is_docket_available():
                    return TaskConfig(mode="optional")
            except Exception:
                pass
        return None

    def _register_resources(self) -> None:
        """Register MCP resources for agent context.

        The documents ship inside the package (``simul_mcp/resources``) so a
        wheel install serves them the same way an editable checkout does.
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
            return f"{'/'.join(parts)} is missing from the simul_mcp package."

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
            description="SimulationContext, PhysicsContext, Articulation, RigidPrim, XFormPrim reference.",
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
        derived = self.settings.isaac_sim.bridge_port + (
            socket_port - self.settings.isaac_sim.socket_port
        )
        return max(1024, min(derived, 65535))

    def _socket_port_for_bridge(self, bridge_port: int) -> int:
        """Derive the VS Code socket port for an instance from its bridge port."""
        derived = self.settings.isaac_sim.socket_port + (
            bridge_port - self.settings.isaac_sim.bridge_port
        )
        return max(1024, min(derived, 65535))

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
        resolved_bridge_port = (
            bridge_port
            if bridge_port is not None
            else self._bridge_port_for_socket(socket_port)
        )
        resolved_bridge_timeout = (
            bridge_timeout
            if bridge_timeout is not None
            else self.settings.isaac_sim.bridge_timeout
        )
        resolved_fallback = (
            bridge_fallback_to_vscode
            if bridge_fallback_to_vscode is not None
            else self.settings.isaac_sim.bridge_fallback_to_vscode
        )
        return IsaacSocketClient(
            host=socket_host,
            port=socket_port,
            bridge_host=bridge_host or socket_host,
            bridge_port=resolved_bridge_port,
            bridge_socket_path=bridge_socket_path,
            bridge_timeout_seconds=resolved_bridge_timeout,
            prefer_bridge=bridge_enabled,
            fallback_to_vscode=resolved_fallback,
            timeout_seconds=socket_timeout,
            socket_protocol=socket_protocol or self.settings.isaac_sim.socket_protocol,
            auth_token=(
                socket_auth_token
                if socket_auth_token is not None
                else self.settings.isaac_sim.socket_auth_token
            ),
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
            except (OSError, json.JSONDecodeError):
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
                    # (/tmp/simul-mcp/...), not the host's. The socket must
                    # live in the discovery dir anyway, so try its basename
                    # inside the local dir — inside the boundary by
                    # construction.
                    resolved = os.path.realpath(
                        os.path.join(discovery_dir, os.path.basename(str(socket_path)))
                    )
                if not resolved.startswith(boundary) or not os.path.exists(resolved):
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

            # Check if PID is still alive
            if isinstance(pid, int):
                try:
                    os.kill(pid, 0)  # signal 0 = check existence
                except ProcessLookupError:
                    # Process is dead -- clean up stale file
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                    continue
                except PermissionError:
                    pass  # Process exists but we can't signal it -- that's fine

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

        candidates: Dict[int, IsaacSocketClient] = {}
        for port in range(scan_start, scan_end):
            if port in existing_ports:
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

    def _backend_enabled(self, name: str) -> bool:
        """Return True if *name* is in the selected backends (or all when None)."""
        return self._backends is None or name in self._backends

    def _register_tools(self) -> None:
        """Register MCP tools for enabled backends only."""
        from .registration import (
            register_blender_tools,
            register_instance_tools,
            register_isaac_tools,
            register_stats_tools,
            register_unreal_tools,
            register_usd_tools,
        )

        # USD file operations (headless, local files only). Registering them
        # without the adapter would expose tools that fail on every call.
        if self._backend_enabled("usd") and self.headless_adapter is not None:
            register_usd_tools(self)

        # Isaac Sim tools (TCP socket to running instance), including the
        # instance discovery / routing tools that only make sense with Isaac.
        if self._backend_enabled("isaac"):
            register_instance_tools(self)
            register_isaac_tools(self)

        # Blender tools (if runtime available)
        if self._backend_enabled("blender"):
            if self.blender_adapter and self.blender_adapter.is_available():
                register_blender_tools(self)

        # Unreal tools. The thin surface (health, ping, instance listing,
        # capture, exec script) keeps the MCP tool list small; the full
        # surface is opted into via unreal.tool_surface / --unreal-tools full.
        if self._backend_enabled("unreal"):
            if self.unreal_adapter and self.unreal_adapter.is_available():
                register_unreal_tools(
                    self, thin=self.settings.unreal.tool_surface == "thin"
                )

        # Usage statistics (always available)
        register_stats_tools(self)

        # FastMCP 3.x stores tools in local_provider._components with
        # keys like "tool:<name>@".  Count entries whose key starts with
        # "tool:" to get an accurate tool count.
        tool_count = 0
        lp = getattr(self.mcp, "local_provider", None)
        if lp is not None:
            components = getattr(lp, "_components", {})
            tool_count = sum(1 for k in components if k.startswith("tool:"))
        self.logger.info(f"Registered {tool_count} MCP tools")

    async def shutdown(self) -> None:
        """Release server resources on shutdown."""
        if self.headless_adapter is not None:
            self.headless_adapter.close()
        self._isaac_clients.clear()
        self._rate_limiters.clear()
        self.usage_tracker._stats.clear()
        self.usage_tracker._recent.clear()
        logger.info("Simul 3D MCP Server shut down")

    async def run(self, transport: str = "stdio") -> None:
        """
        Run the MCP server.

        Args:
            transport: Transport type (stdio, sse)
        """
        try:
            self.logger.info(f"Starting Simul 3D MCP Server with {transport} transport")

            if transport == "stdio":
                await self.mcp.run_async(transport="stdio")
            elif transport == "sse":
                await self.mcp.run_async(
                    transport="sse",
                    host=self.settings.server.host,
                    port=self.settings.server.port,
                )
            else:
                raise ValueError(f"Unsupported transport: {transport}")

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
            Mapping of backend name (``isaac``, ``usd``, ``blender``,
            ``unreal``) to ``{"enabled": bool, "available": bool,
            "capabilities": list[str]}``. ``enabled`` is whether the backend
            was selected for tool registration.
        """
        isaac_transports: List[str] = ["socket"]
        if self.client.bridge_enabled:
            isaac_transports.append("bridge")

        return {
            "isaac": {
                "enabled": self._backend_enabled("isaac"),
                "available": True,
                "capabilities": isaac_transports,
            },
            "usd": self._adapter_capabilities("usd", self.headless_adapter),
            "blender": self._adapter_capabilities("blender", self.blender_adapter),
            "unreal": self._adapter_capabilities("unreal", self.unreal_adapter),
        }

    def _adapter_capabilities(
        self, backend: str, adapter: Optional[Any]
    ) -> Dict[str, Any]:
        """Build one ``get_capabilities`` entry for an adapter-backed backend."""
        available = adapter is not None and adapter.is_available()
        return {
            "enabled": self._backend_enabled(backend),
            "available": available,
            "capabilities": sorted(adapter.get_capabilities()) if available else [],
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
