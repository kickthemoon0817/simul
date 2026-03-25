"""
Simul 3D MCP Server implementation.

This module provides the main MCP server class with tool registry,
connection management, and 3D simulation/DCC integration based on FastMCP.
"""

import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel

try:
    from fastmcp import FastMCP
    from fastmcp.server.tasks import TaskConfig
    from mcp.types import (
        EmbeddedResource,
        ImageContent,
        TextContent,
        Tool,
        ToolAnnotations,
    )

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCP = None
    Tool = None
    TextContent = None
    ImageContent = None
    EmbeddedResource = None
    ToolAnnotations = None
    TaskConfig = None

from ..adapters import (
    BlenderRuntimeAdapter,
    HeadlessUSDAdapter,
    IsaacSocketClient,
    is_blender_available,
    is_headless_available,
)
from ..config import Settings, get_settings
from ..logging import LoggerMixin, get_logger
from ..utils.timing import RateLimiter
from .schemas import *
from .tools.isaac_tools import IsaacTools
from .usage_tracker import ToolUsageTracker

logger = get_logger(__name__)

try:
    from importlib.metadata import version as _pkg_version

    _PACKAGE_VERSION: str = _pkg_version("simul-mcp")
except Exception:
    _PACKAGE_VERSION = "0.0.6"

_MCP_INSTRUCTIONS: str = (
    "Simul MCP provides tools for interacting with 3D simulation "
    "and DCC applications. Use Isaac Sim tools to control a running NVIDIA Isaac Sim "
    "instance — granular tools for scene inspection, prim "
    "manipulation, physics, simulation control, materials, "
    "viewport/camera, and asset/stage operations. The generic "
    "execute_isaac_script tool is available for custom scripts. "
    "Use USD tools to load, inspect, and edit Universal Scene "
    "Description files. "
    "Use Blender tools when a Blender runtime is connected. "
    "Use Unreal tools when an Unreal Engine instance is connected.\n\n"
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
    "  2. Call set_active_isaac_instance to switch to the correct one.\n"
    "  3. All subsequent isaac_* calls route to that instance.\n\n"
    "When working with a live Isaac Sim session, prefer isaac_* tools. "
    "Use execute_isaac_script for operations not covered by granular tools. "
    "Read the 'simul://isaac-sim/skills' resource for scripting patterns "
    "and API reference before writing execute_isaac_script code."
)

_FASTMCP_SUPPORTS_INSTRUCTIONS: bool = (
    "instructions" in inspect.signature(FastMCP).parameters
    if FASTMCP_AVAILABLE and FastMCP is not None
    else False
)


class SimulMCPServer(LoggerMixin):
    """
    Simul 3D MCP Server.

    Unified MCP server for 3D simulation and DCC tools. Registers tools for
    Isaac Sim (live TCP), headless USD, Blender, and Unreal Engine backends.
    Supports multi-instance Isaac Sim discovery and active instance routing.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Simul 3D MCP Server.

        Args:
            settings: Configuration settings
        """
        if not FASTMCP_AVAILABLE:
            raise ImportError("FastMCP not available. Please install fastmcp package.")

        self.settings = settings or get_settings()
        self._project_root = Path(__file__).resolve().parents[3]
        self._allowed_paths = self._resolve_allowed_paths()

        self.usage_tracker = ToolUsageTracker()
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
        default_client = IsaacSocketClient(
            host=self.settings.isaac_sim.socket_host,
            port=self.settings.isaac_sim.socket_port,
            timeout_seconds=self.settings.isaac_sim.socket_timeout,
        )
        self._isaac_clients["default"] = default_client
        for inst in self.settings.isaac_sim.instances:
            self._isaac_clients[inst.name] = IsaacSocketClient(
                host=inst.host,
                port=inst.port,
                timeout_seconds=inst.timeout,
            )
        self.client = default_client
        self._isaac_tools = IsaacTools(self.client, self.settings)

        self.blender_adapter = (
            BlenderRuntimeAdapter(self.settings) if is_blender_available() else None
        )

        # Initialize FastMCP server
        assert FastMCP is not None

        mcp_kwargs: Dict[str, Any] = {
            "name": "Simul – 3D Simulation & DCC Tools",
            "version": _PACKAGE_VERSION,
        }
        if _FASTMCP_SUPPORTS_INSTRUCTIONS:
            mcp_kwargs["instructions"] = _MCP_INSTRUCTIONS

        self.mcp = FastMCP(
            **mcp_kwargs,
        )

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
            ).dict()
        return None

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
        t0 = time.monotonic()
        try:
            result = await coro
            duration_ms = (time.monotonic() - t0) * 1000
            success = not result.get("error")
            self.usage_tracker.record(
                tool_name, duration_ms, success, params=params,
                error=result.get("error") if not success else None,
            )
            return result
        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            self.usage_tracker.record(
                tool_name, duration_ms, False, params=params, error=str(exc),
            )
            logger.error("Isaac tool %s failed: %s", tool_name, exc)
            return ErrorResponse(
                error=str(exc), error_type=type(exc).__name__
            ).dict()

    def _resolve_allowed_paths(self) -> List[Path]:
        allowed_paths: List[Path] = []
        for path_str in self.settings.security.allowed_paths:
            expanded = os.path.expandvars(path_str)
            candidate = Path(expanded).expanduser()
            if not candidate.is_absolute():
                candidate = self._project_root / candidate
            try:
                candidate = candidate.resolve()
            except Exception:
                candidate = candidate.absolute()
            allowed_paths.append(candidate)
        return allowed_paths

    def _is_path_allowed(self, path_str: str) -> bool:
        if not self.settings.security.sandbox_enabled:
            return True
        if not path_str:
            return False
        expanded = os.path.expandvars(path_str)
        candidate = Path(expanded).expanduser()
        if not candidate.is_absolute():
            candidate = self._project_root / candidate
        try:
            candidate = candidate.resolve()
        except Exception:
            candidate = candidate.absolute()
        for allowed_path in self._allowed_paths:
            try:
                candidate.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False

    def _validate_input(
        self, model: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, Dict[str, Any]]:
        try:
            return model(**kwargs)
        except Exception as e:
            return ErrorResponse(
                error=str(e), error_type="ValidationError", details={"input": kwargs}
            ).dict()

    def _validate_output(
        self,
        result: Any,
        models: Tuple[Type[BaseModel], ...],
        tool_name: str,
    ) -> Dict[str, Any]:
        if isinstance(result, BaseModel):
            payload: Any = result.dict()
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
            ).dict()

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
        ).dict()

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

        @self.mcp.resource(
            "simul://isaac-sim/skills",
            name="Isaac Sim Scripting Skills",
            description=(
                "Isaac Sim 5.1.0 scripting reference for execute_isaac_script: "
                "API patterns, code templates, namespace migration notes, "
                "and a quick-reference table for common operations."
            ),
        )
        def isaac_sim_skills() -> str:
            if skills_path.is_file():
                return skills_path.read_text(encoding="utf-8")
            return "skills.md not found at project root."

    def _switch_active_instance(self, name: str) -> None:
        """
        Switch the active Isaac Sim instance by name.

        Updates both ``self.client`` (used by execute_isaac_script / ping)
        and ``self._isaac_tools._client`` (used by all granular tools).

        Args:
            name: Instance name key in ``_isaac_clients``.
        """
        client = self._isaac_clients[name]
        self._active_instance = name
        self.client = client
        self._isaac_tools._client = client

    async def _scan_isaac_instances(self) -> Dict[str, IsaacSocketClient]:
        """
        Scan the configured port range for running Isaac Sim instances.

        Returns any newly discovered instances not already in the registry.
        Existing named instances are never overwritten.
        """
        scan_start = self.settings.isaac_sim.scan_port_start
        scan_end = self.settings.isaac_sim.scan_port_end
        host = self.settings.isaac_sim.socket_host
        timeout = self.settings.isaac_sim.socket_timeout
        existing_ports = {c._port for c in self._isaac_clients.values() if c._host == host}

        discovered: Dict[str, IsaacSocketClient] = {}
        for port in range(scan_start, scan_end):
            if port in existing_ports:
                continue
            candidate = IsaacSocketClient(host=host, port=port, timeout_seconds=min(timeout, 3.0))
            if await candidate.ping():
                name = f"isaac-{port}"
                discovered[name] = candidate
        return discovered

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
        info: Dict[str, Any] = {
            "name": name,
            "host": client._host,
            "port": client._port,
            "reachable": False,
            "active": name == self._active_instance,
            "stage_url": None,
            "up_axis": None,
            "prim_count": None,
            "is_playing": None,
        }
        try:
            result = await client.execute(
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
                # Reachable but script failed — still mark as reachable
                info["reachable"] = True
        except (ConnectionRefusedError, TimeoutError, OSError):
            info["reachable"] = False
        except (json.JSONDecodeError, KeyError):
            info["reachable"] = True
        return info


    def _register_tools(self) -> None:
        """Register all MCP tools."""
        from .registration import (
            register_instance_tools,
            register_usd_tools,
            register_isaac_tools,
            register_blender_tools,
            register_unreal_tools,
            register_stats_tools,
        )

        # Instance discovery and routing (must be first)
        register_instance_tools(self)

        # USD file operations (headless, local files only)
        register_usd_tools(self)

        # Isaac Sim tools (TCP socket to running instance)
        register_isaac_tools(self)

        # Blender tools (if runtime available)
        if self.blender_adapter and self.blender_adapter.is_available():
            register_blender_tools(self)

        # Unreal tools (independent of Blender availability)
        from ..adapters import is_unreal_available

        if is_unreal_available():
            register_unreal_tools(self)

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

    def get_capabilities(self) -> List[str]:
        """Get list of server capabilities."""
        capabilities = []

        if self.headless_adapter and self.headless_adapter.is_available():
            capabilities.extend(self.headless_adapter.get_capabilities())

        if self.blender_adapter and self.blender_adapter.is_available():
            capabilities.extend(self.blender_adapter.get_capabilities())

        return list(set(capabilities))  # Remove duplicates


# Convenience functions
def create_server_instance(settings: Optional[Settings] = None) -> SimulMCPServer:
    """
    Create a Simul 3D MCP Server instance.

    Args:
        settings: Configuration settings

    Returns:
        SimulMCPServer instance
    """
    return SimulMCPServer(settings)


async def start_mcp_server(
    settings: Optional[Settings] = None, transport: str = "stdio"
) -> None:
    """
    Start the Simul 3D MCP Server.

    Args:
        settings: Configuration settings
        transport: Transport type
    """
    server = create_server_instance(settings)
    await server.run(transport)
