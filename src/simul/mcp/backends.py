"""The backend registry the MCP server iterates.

One ``BackendSpec`` per backend names everything the server needs to wire it
up: how to build its adapter from settings, which registration function adds
its tools, which settings section configures it, what the server's
instructions say about it, and any gate on its tools beyond the runtime
being present. Adding a backend is one entry here plus an adapter
implementing ``BackendAdapter``.

The availability probes and adapter classes are module attributes on
purpose: a test that wants a runtime to look present or absent patches them
here, in the one place the server reads them from. The probes cover the
runtime (``is_headless_available``, ``is_blender_available``,
``is_unreal_available``) and, for Isaac Sim, the operating system
(``is_isaac_platform_supported``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, AbstractSet, Callable, Optional, Tuple

from ..adapters import (
    BlenderRuntimeAdapter,
    HeadlessUSDAdapter,
    IsaacRuntimeAdapter,
    UnrealRuntimeAdapter,
    is_blender_available,
    is_headless_available,
    is_unreal_available,
)
from ..adapters.base import BackendAdapter
from ..config import Settings
from .registration import (
    register_blender_tools,
    register_instance_tools,
    register_isaac_tools,
    register_unreal_tools,
    register_usd_tools,
)

if TYPE_CHECKING:
    from .server import SimulMCPServer

AdapterFactory = Callable[[Settings], Optional[BackendAdapter]]
ToolRegistrar = Callable[["SimulMCPServer"], None]
# (settings, named in --backends) -> why the tools stay unregistered, or None.
SkipCheck = Callable[[Settings, bool], Optional[str]]

#: Skip reason of a backend that a ``--backends`` selection left out.
NOT_SELECTED = "left out of --backends"

#: Skip reason of the Isaac tools on an OS Isaac Sim does not run on.
ISAAC_PLATFORM_SKIP = (
    "Isaac Sim needs Linux or Windows; to register its tools anyway (an unsupported setup, "
    "such as an SSH tunnel to an Isaac Sim host) set ISAAC_SIM__ENABLED=true or pass --backends isaac"
)

#: Skip reason of the Isaac tools when the configuration switches them off.
ISAAC_DISABLED_SKIP = (
    "isaac_sim.enabled is false (ISAAC_SIM__ENABLED); unset it or set it to true to "
    "register the Isaac Sim tools"
)


@dataclass(frozen=True)
class BackendSpec:
    """Everything the server needs to know about one backend.

    Attributes:
        name: Registry name, the value accepted by ``--backends`` and the
            token the routing rule keys on.
        label: Human-readable name for the CLI's tool grouping.
        settings_attribute: Attribute of ``Settings`` holding the backend's
            configuration section.
        adapter_factory: Builds the adapter from settings, or returns None
            when the runtime is not importable in this process.
        register_tools: Adds the backend's tools to the server.
        routing_rule: The line the ROUTING instructions state for the
            backend, phrased in terms of the token inside tool names.
        summary: The sentence that introduces the backend's tools at the
            top of the instructions.
        guidance: Further instructions for the backend's tools as whole
            paragraphs, stated only when they are registered; empty when the
            summary and the routing rule say enough.
        skip_check: A gate on the tools beyond the runtime being present.
            Called with the settings and whether ``--backends`` named the
            backend, it returns why the tools stay unregistered, or None to
            register them. None means the backend has no such gate.
    """

    name: str
    label: str
    settings_attribute: str
    adapter_factory: AdapterFactory
    register_tools: ToolRegistrar
    routing_rule: str
    summary: str
    guidance: str = ""
    skip_check: Optional[SkipCheck] = None


def is_isaac_platform_supported(platform: Optional[str] = None) -> bool:
    """Report whether Isaac Sim runs on this operating system.

    NVIDIA ships Isaac Sim for Linux and Windows only, so elsewhere (macOS)
    no Isaac Sim on this machine can answer the Isaac tools.

    Args:
        platform: A ``sys.platform`` value; None reads this interpreter's.

    Returns:
        True on Linux and on Windows (Cygwin included).
    """
    name = sys.platform if platform is None else platform
    return name.startswith("linux") or name in ("win32", "cygwin")


def _isaac_skip_check(settings: Settings, named: bool) -> Optional[str]:
    """Gate the Isaac tools on the platform unless the user decided.

    ``isaac_sim.enabled`` decides when it is set. Unset, the tools register
    when ``--backends`` names Isaac or this OS runs Isaac Sim.

    Args:
        settings: Server settings.
        named: Whether ``--backends`` named the backend.

    Returns:
        Why the tools stay unregistered, or None to register them.
    """
    enabled = settings.isaac_sim.enabled
    if enabled is False:
        return ISAAC_DISABLED_SKIP
    if enabled or named or is_isaac_platform_supported():
        return None
    return ISAAC_PLATFORM_SKIP


def _headless_adapter(settings: Settings) -> Optional[BackendAdapter]:
    """Build the headless USD adapter when pxr imports."""
    return HeadlessUSDAdapter(settings) if is_headless_available() else None


def _blender_adapter(settings: Settings) -> Optional[BackendAdapter]:
    """Attached mode does not require bpy in the MCP server interpreter."""
    if settings.blender.mode == "attached" or is_blender_available():
        return BlenderRuntimeAdapter(settings)
    return None


def _unreal_adapter(settings: Settings) -> Optional[BackendAdapter]:
    """Build the Unreal adapter when its module imported and aiohttp (or the embedded module) is present."""
    if UnrealRuntimeAdapter is None or not is_unreal_available():
        return None
    return UnrealRuntimeAdapter(settings)


def _register_isaac(server: "SimulMCPServer") -> None:
    """Register the Isaac tools plus the instance routing tools and scripting references that serve them."""
    register_instance_tools(server)
    register_isaac_tools(server)
    server._register_isaac_resources()


def _register_unreal(server: "SimulMCPServer") -> None:
    """Register the thin or the full Unreal surface, as ``unreal.tool_surface`` says."""
    register_unreal_tools(server, thin=server.settings.unreal.tool_surface == "thin")


def _register_blender(server: "SimulMCPServer") -> None:
    """Register the thin or the full Blender surface, as ``blender.tool_surface`` says."""
    register_blender_tools(server, thin=server.settings.blender.tool_surface == "thin")


_ISAAC_GUIDANCE = (
    "ISAAC SIM — for GUI/app state (windows, focus, selection, timeline) use "
    "get_isaac_ui_state; to inspect one window's widget tree use get_isaac_ui_window. "
    "Reach for execute_isaac_script when no granular tool covers the operation (custom "
    "extensions, replicator workflows, robotics APIs, warp kernels). Read the "
    "'simul://isaac-sim/skills' resource for scripting patterns and API reference when "
    "writing scripts.\n"
    "  Finding prims: get_isaac_scene_summary for counts first; list_isaac_prims to "
    "browse a level; get_isaac_subtree for the structure under one prim; "
    "search_isaac_prims by name or exact type anywhere below a root; "
    "query_isaac_typed_prims by schema (subclasses too) with attribute values; "
    "find_isaac_prims_in_area by position.\n"
    "  Conventions: Isaac Sim stages are Z-up with metersPerUnit=1, so positions and "
    "distances are metres, gravity points along -Z, and rotations are XYZ Euler degrees; "
    "get_isaac_stage_info reports the up_axis and meters_per_unit of a loaded file when "
    "they differ.\n"
    "  Multiple instances: list_isaac_instances, then set_active_isaac_instance; later "
    "Isaac calls in this session go there. For containerized Isaac Sim use the "
    "host-published ports, not the container-internal ones."
)

BACKENDS: Tuple[BackendSpec, ...] = (
    BackendSpec(
        name="isaac",
        label="Isaac Sim",
        settings_attribute="isaac_sim",
        adapter_factory=IsaacRuntimeAdapter,
        register_tools=_register_isaac,
        routing_rule="Tools containing 'isaac' → require a running Isaac Sim instance (TCP socket).",
        summary=(
            "Use Isaac Sim tools to control a running NVIDIA Isaac Sim instance — granular "
            "tools for scene inspection, prim manipulation, physics, simulation control, "
            "materials, viewport/camera, rendering, and asset/stage operations."
        ),
        guidance=_ISAAC_GUIDANCE,
        skip_check=_isaac_skip_check,
    ),
    BackendSpec(
        name="usd",
        label="USD / Headless",
        settings_attribute="usd",
        adapter_factory=_headless_adapter,
        register_tools=register_usd_tools,
        routing_rule=(
            "Tools containing none of those names (load_usd_file, get_prim_info, "
            "create_prim, summarize_scene, etc.) → operate on local USD files via the "
            "headless adapter; they do NOT connect to any engine."
        ),
        summary="Use USD tools to load, inspect, and edit Universal Scene Description files.",
    ),
    BackendSpec(
        name="blender",
        label="Blender",
        settings_attribute="blender",
        adapter_factory=_blender_adapter,
        register_tools=_register_blender,
        routing_rule=(
            "Tools containing 'blender' or 'simready' → use local bpy in embedded mode, or the exact "
            "window selected with 'simul blender attach' in attached mode; never switch targets automatically."
        ),
        summary="Use Blender tools when a Blender runtime is connected.",
    ),
    BackendSpec(
        name="unreal",
        label="Unreal",
        settings_attribute="unreal",
        adapter_factory=_unreal_adapter,
        register_tools=_register_unreal,
        routing_rule=(
            "Tools containing 'unreal' → require a connected Unreal Engine instance. In attached mode use the "
            "editor/map/viewport selected with 'simul unreal attach'; never switch targets automatically. "
            "control_unreal_ui cursors are virtual overlays, not OS mouse input."
        ),
        summary="Use Unreal tools when an Unreal Engine instance is connected.",
    ),
)

ALL_BACKEND_NAMES: frozenset[str] = frozenset(spec.name for spec in BACKENDS)


def backend_spec(name: str) -> BackendSpec:
    """Return the spec registered under ``name``.

    Args:
        name: A backend's registry name.

    Returns:
        The matching spec.

    Raises:
        KeyError: When no backend is registered under ``name``.
    """
    for spec in BACKENDS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def backend_skip_reason(
    name: str, settings: Settings, selected: Optional[AbstractSet[str]] = None
) -> Optional[str]:
    """Explain why a backend registers no tools whatever its runtime.

    The server and the CLI's startup banner both ask this, so they agree on
    what is skipped and why.

    Args:
        name: A backend's registry name.
        settings: Server settings.
        selected: The ``--backends`` selection; None selects every backend.

    Returns:
        ``NOT_SELECTED`` for a backend the selection left out, the reason
        the backend's ``skip_check`` gives (the Isaac platform check), or
        None when the tools register as long as the runtime is available.

    Raises:
        KeyError: When no backend is registered under ``name``.
    """
    spec = backend_spec(name)
    if selected is not None and name not in selected:
        return NOT_SELECTED
    if spec.skip_check is None:
        return None
    return spec.skip_check(settings, selected is not None)
