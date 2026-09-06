"""The backend registry the MCP server iterates.

One ``BackendSpec`` per backend names everything the server needs to wire it
up: how to build its adapter from settings, which registration function adds
its tools, which settings section configures it, and the routing rule the
server's instructions state for it. Adding a backend is one entry here plus
an adapter implementing ``BackendAdapter``.

The availability probes and adapter classes are module attributes on
purpose: a test that wants a runtime to look present or absent patches them
here, in the one place the server reads them from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple

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
    """

    name: str
    label: str
    settings_attribute: str
    adapter_factory: AdapterFactory
    register_tools: ToolRegistrar
    routing_rule: str


def _headless_adapter(settings: Settings) -> Optional[BackendAdapter]:
    """Build the headless USD adapter when pxr imports."""
    return HeadlessUSDAdapter(settings) if is_headless_available() else None


def _blender_adapter(settings: Settings) -> Optional[BackendAdapter]:
    """Build the Blender adapter when bpy imports."""
    return BlenderRuntimeAdapter(settings) if is_blender_available() else None


def _unreal_adapter(settings: Settings) -> Optional[BackendAdapter]:
    """Build the Unreal adapter when its module imported and aiohttp (or the embedded module) is present."""
    if UnrealRuntimeAdapter is None or not is_unreal_available():
        return None
    return UnrealRuntimeAdapter(settings)


def _register_isaac(server: "SimulMCPServer") -> None:
    """Register the Isaac tools and the instance routing tools that only make sense with them."""
    register_instance_tools(server)
    register_isaac_tools(server)


def _register_unreal(server: "SimulMCPServer") -> None:
    """Register the thin or the full Unreal surface, as ``unreal.tool_surface`` says."""
    register_unreal_tools(server, thin=server.settings.unreal.tool_surface == "thin")


BACKENDS: Tuple[BackendSpec, ...] = (
    BackendSpec(
        name="isaac",
        label="Isaac Sim",
        settings_attribute="isaac_sim",
        adapter_factory=IsaacRuntimeAdapter,
        register_tools=_register_isaac,
        routing_rule="Tools containing 'isaac' → require a running Isaac Sim instance (TCP socket).",
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
    ),
    BackendSpec(
        name="blender",
        label="Blender",
        settings_attribute="blender",
        adapter_factory=_blender_adapter,
        register_tools=register_blender_tools,
        routing_rule="Tools containing 'blender' → require a connected Blender runtime.",
    ),
    BackendSpec(
        name="unreal",
        label="Unreal",
        settings_attribute="unreal",
        adapter_factory=_unreal_adapter,
        register_tools=_register_unreal,
        routing_rule="Tools containing 'unreal' → require a connected Unreal Engine instance.",
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
