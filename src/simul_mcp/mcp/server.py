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
from typing import Any, Coroutine, Dict, List, Optional, Set, Tuple, Type, Union

from fastmcp import FastMCP
from fastmcp.server.context import _current_context
from fastmcp.server.tasks import TaskConfig
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from ..adapters import (
    BlenderRuntimeAdapter,
    HeadlessUSDAdapter,
    IsaacSocketClient,
    UnrealRuntimeAdapter,
    is_blender_available,
    is_headless_available,
    is_unreal_available,
)
from .. import __version__ as _source_version
from ..config import Settings, get_settings
from ..logging import LoggerMixin, get_logger
from ..utils.paths import PathPolicy
from ..utils.timing import RateLimiter
from .schemas.common import ErrorResponse
from .tools.isaac_tools import IsaacTools
from .session_manager import SessionManager
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
    "TOOL SELECTION — check granular tools first, use execute_isaac_script "
    "when no granular tool covers the operation:\n"
    "  - RTX/renderer settings → get/set_isaac_carb_settings\n"
    "  - AOV/render pass reads → read_isaac_aovs (full pipeline in one call)\n"
    "  - Available AOVs → list_isaac_aovs\n"
    "  - Find prims by type + read attributes → query_isaac_typed_prims\n"
    "  - Viewport state → get_isaac_viewport_info\n"
    "  - Render variables → list_isaac_render_vars\n"
    "  - Scene inspection → get_isaac_prim_info, list_isaac_prims, search_isaac_prims\n"
    "  - Physics → get_isaac_rigid_body_info, get_isaac_collision_info, etc.\n"
    "  - Materials → get_isaac_material_info, create_isaac_material, etc.\n"
    "  - Simulation → start/stop/step_isaac_simulation\n"
    "  - Camera → set_isaac_camera, capture_isaac_viewport\n"
    "For operations not covered above (custom extensions, advanced replicator "
    "workflows, robotics APIs, warp kernels, etc.), use execute_isaac_script "
    "freely. Read the 'simul://isaac-sim/skills' resource for scripting "
    "patterns and API reference when writing scripts.\n\n"
    "ROUTING — tool name prefixes determine the backend:\n"
    "  isaac_* tools → require a running Isaac Sim instance (TCP socket).\n"
    "  Non-prefixed USD tools (load_usd_file, get_prim_info, create_prim, "
    "etc.) → operate on local USD files via the headless adapter; "
    "they do NOT connect to Isaac Sim.\n"
    "  blender_* tools → require a connected Blender runtime.\n"
    "  unreal_* tools → require a connected Unreal Engine instance.\n\n"
    "MULTI-INSTANCE — when multiple Isaac Sim applications are running:\n"
    "  1. Call list_isaac_instances to discover all running instances "
    "and see which stage each has loaded.\n"
    "  2. Call set_active_isaac_instance to switch within the current MCP session.\n"
    "  3. All subsequent isaac_* calls in that same session route to that instance.\n"
    "  4. For containerized Isaac Sim, use the host-published bridge / VS Code ports, "
    "not the container-internal ports."
)

_FASTMCP_SUPPORTS_INSTRUCTIONS: bool = (
    "instructions" in inspect.signature(FastMCP).parameters
)


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
        self._project_root = Path(__file__).resolve().parents[3]
        self._path_policy = PathPolicy.from_settings(
            self.settings, project_root=self._project_root
        )
        self._allowed_paths = self._resolve_allowed_paths()

        self.usage_tracker = ToolUsageTracker()
        self.session_manager = SessionManager()
        self._session_routes: Dict[str, IsaacSessionRoute] = {}
        self._isaac_instance_locks: Dict[str, asyncio.Lock] = {}
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._rate_limit_enabled = self.settings.security.rate_limiting_enabled
        self._rate_limit_rate = self.settings.security.requests_per_minute / 60.0
        self._rate_limit_burst = self.settings.security.burst_size
        self._tool_timeout = self.settings.server.timeout

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
            UnrealRuntimeAdapter(self.settings) if UnrealRuntimeAdapter is not None else None
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

    def _get_rate_limiter(self, tool_name: str) -> Optional[RateLimiter]:
        if not self._rate_limit_enabled:
            return None

        if tool_name not in self._rate_limiters:
            self._rate_limiters[tool_name] = RateLimiter(
                self._rate_limit_rate,
                self._rate_limit_burst,
            )
        return self._rate_limiters[tool_name]

    def _check_rate_limit(self, tool_name: str) -> Optional[Dict[str, Any]]:
        limiter = self._get_rate_limiter(tool_name)
        if limiter and not limiter.acquire():
            return ErrorResponse(
                error="Rate limit exceeded",
                error_type="RateLimitError",
                details={"tool": tool_name},
            ).model_dump()
        return None

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

    async def _exec_isaac(
        self,
        tool_name: str,
        coro: Coroutine[Any, Any, Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute an Isaac Sim tool coroutine with rate limiting,
        unified error handling, and usage tracking.

        Args:
            tool_name: Name of the tool for rate limiting.
            coro: Awaitable coroutine returned by an IsaacTools method.
            params: Optional dict of call parameters for usage logging.

        Returns:
            Tool result dict or error response dict.
        """
        rate_error = self._check_rate_limit(tool_name)
        if rate_error is not None:
            self.usage_tracker.record(
                tool_name, 0.0, False, params=params, error="rate_limited",
            )
            return rate_error
        instance_name = self._get_effective_instance_name()
        lock = self._get_instance_lock(instance_name)
        async with lock:
            t0 = time.monotonic()
            try:
                result = await coro
                duration_ms = (time.monotonic() - t0) * 1000
                success = not result.get("error")
                self.usage_tracker.record(
                    tool_name, duration_ms, success, params=params,
                    error=result.get("error") if not success else None,
                )
                binding = self._get_active_binding()
                if binding is not None:
                    self.session_manager.get_instance_session(
                        binding.port
                    ).heartbeat(binding.agent_id, tool_name)
                    binding.last_heartbeat = time.time()
                return result
            except Exception as exc:
                duration_ms = (time.monotonic() - t0) * 1000
                self.usage_tracker.record(
                    tool_name, duration_ms, False, params=params, error=str(exc),
                )
                logger.error("Isaac tool %s failed: %s", tool_name, exc)
                return ErrorResponse(
                    error=str(exc), error_type=type(exc).__name__
                ).model_dump()

    def _resolve_allowed_paths(self) -> List[Path]:
        return self._path_policy.allowed_roots

    def _is_path_allowed(self, path_str: str) -> bool:
        # Kept as a fast-fail at the MCP boundary. The authoritative check now
        # lives in the tools layer, below both this and the CLI.
        return self._path_policy.is_allowed(path_str)

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

    def _tool_annotations(
        self,
        read_only: bool,
        idempotent: bool,
        open_world: bool,
        destructive: bool = False,
    ) -> Optional[Any]:
        annotations = {
            "readOnlyHint": read_only,
            "idempotentHint": idempotent,
            "openWorldHint": open_world,
            "destructiveHint": destructive,
        }
        if ToolAnnotations:
            return ToolAnnotations(**annotations)
        return annotations

    def _tool_output_schema(self, *models: Type[BaseModel]) -> Dict[str, Any]:
        if len(models) == 1:
            model = models[0]
            if hasattr(model, "model_json_schema"):
                return model.model_json_schema()
            return model.schema()

        # FastMCP 2.x validates output schema as a single object and rejects
        # union schemas such as oneOf. Use a permissive object schema for
        # multi-response tools to stay compatible across FastMCP versions.
        return {
            "type": "object",
            "additionalProperties": True,
        }

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
        """Register MCP resources for agent context."""
        skills_path = self._project_root / "skills.md"
        docs_api_dir = self._project_root / "docs" / "api"
        resource = getattr(self.mcp, "resource", None)
        if not callable(resource):
            self.logger.debug(
                "FastMCP resource API unavailable; skipping resource registration"
            )
            return

        @resource(
            "simul://isaac-sim/skills",
            name="Isaac Sim Scripting Skills",
            description=(
                "Isaac Sim 5.1.0 scripting reference: API patterns, "
                "namespace migration notes, and quick-reference table. "
                "Only consult this when no granular tool exists for your task."
            ),
        )
        def isaac_sim_skills() -> str:
            if skills_path.is_file():
                return skills_path.read_text(encoding="utf-8")
            return "skills.md not found at project root."

        def _make_api_reader(fpath: Path) -> str:
            """Read an API reference doc from docs/api/."""
            if fpath.is_file():
                return fpath.read_text(encoding="utf-8")
            return f"{fpath.name} not found."

        @resource(
            "simul://isaac-sim/api/core",
            name="Isaac Sim Core API",
            description="SimulationContext, PhysicsContext, Articulation, RigidPrim, XFormPrim reference.",
        )
        def api_core() -> str:
            return _make_api_reader(docs_api_dir / "core.md")

        @resource(
            "simul://isaac-sim/api/sensors",
            name="Isaac Sim Sensors API",
            description="Camera, IMU, Contact, LiDAR (PhysX/RTX), Proximity sensor reference.",
        )
        def api_sensors() -> str:
            return _make_api_reader(docs_api_dir / "sensors.md")

        @resource(
            "simul://isaac-sim/api/physics",
            name="Isaac Sim Physics API",
            description="PhysX interface, tensor API, collision queries, CCT, vehicle physics reference.",
        )
        def api_physics() -> str:
            return _make_api_reader(docs_api_dir / "physics.md")

        @resource(
            "simul://isaac-sim/api/replicator",
            name="Isaac Sim Replicator API",
            description="Annotators, Writers, Orchestrator, domain randomization reference.",
        )
        def api_replicator() -> str:
            return _make_api_reader(docs_api_dir / "replicator.md")

        @resource(
            "simul://isaac-sim/api/robots",
            name="Isaac Sim Robots API",
            description="Manipulators, grippers, IK, motion planning, wheeled robots reference.",
        )
        def api_robots() -> str:
            return _make_api_reader(docs_api_dir / "robots.md")

        @resource(
            "simul://isaac-sim/api/rendering",
            name="Isaac Sim Rendering API",
            description="Viewport, HydraTexture, RTX post-processing, capture reference.",
        )
        def api_rendering() -> str:
            return _make_api_reader(docs_api_dir / "rendering.md")

        @resource(
            "simul://isaac-sim/api/assets",
            name="Isaac Sim Assets API",
            description="URDF/MJCF import, Cloner, OmniGraph nodes reference.",
        )
        def api_assets() -> str:
            return _make_api_reader(docs_api_dir / "assets.md")

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
        derived = (
            self.settings.isaac_sim.bridge_port
            + (socket_port - self.settings.isaac_sim.socket_port)
        )
        return max(1024, min(derived, 65535))

    def _socket_port_for_bridge(self, bridge_port: int) -> int:
        """Derive the VS Code socket port for an instance from its bridge port."""
        derived = (
            self.settings.isaac_sim.socket_port
            + (bridge_port - self.settings.isaac_sim.bridge_port)
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
            bridge_timeout_seconds=resolved_bridge_timeout,
            prefer_bridge=bridge_enabled,
            fallback_to_vscode=resolved_fallback,
            timeout_seconds=socket_timeout,
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

        existing_ports: set[int] = set()
        for c in self._isaac_clients.values():
            existing_ports.add(c._port)  # vscode port
            if c._bridge_configured and c._bridge_port is not None:
                existing_ports.add(c._bridge_port)  # bridge port

        discovered: Dict[str, IsaacSocketClient] = {}
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
            )

            # Verify the instance is actually reachable
            if await client.ping():
                name = f"isaac-{port}"
                discovered[name] = client

        return discovered

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
                stage_info = await client.bridge_request("get_stage_info", {})
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
                "    'prim_count': len(list(stage.Traverse())) if stage else 0,\n"
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
            register_instance_tools,
            register_usd_tools,
            register_isaac_tools,
            register_blender_tools,
            register_unreal_tools,
            register_stats_tools,
        )

        # Instance discovery and routing (always registered)
        register_instance_tools(self)

        # USD file operations (headless, local files only)
        if self._backend_enabled("usd"):
            register_usd_tools(self)

        # Isaac Sim tools (TCP socket to running instance)
        if self._backend_enabled("isaac"):
            register_isaac_tools(self)

        # Blender tools (if runtime available)
        if self._backend_enabled("blender"):
            if self.blender_adapter and self.blender_adapter.is_available():
                register_blender_tools(self)

        # Unreal tools — thin MCP set (health, capture, exec script).
        # Full operations available via CLI: simul unreal --help
        if self._backend_enabled("unreal"):
            if self.unreal_adapter and self.unreal_adapter.is_available():
                register_unreal_tools(self, thin=True)

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

    def get_capabilities(self) -> List[str]:
        """Get list of server capabilities."""
        capabilities = []

        if self.headless_adapter and self.headless_adapter.is_available():
            capabilities.extend(self.headless_adapter.get_capabilities())

        if self.blender_adapter and self.blender_adapter.is_available():
            capabilities.extend(self.blender_adapter.get_capabilities())

        return list(set(capabilities))  # Remove duplicates


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
