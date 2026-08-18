"""Regression: the registered tool surface matches what ships around it.

Two guards born from the 0.1.0 alias removal:

* the fourteen single-``prim_path`` getters folded into
  ``get_isaac_prim_detail`` must stay unregistered — reintroducing one
  silently re-inflates the tool listing;
* every ``mcp__simul__<name>`` reference in the shipped ``skills/`` tree
  must resolve to a live registration, so a skill can never instruct an
  agent to call a tool that no longer exists.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Set

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402

# The aliases 0.1.0 removed. get_isaac_texture_dependencies is deliberately
# absent: it is a subtree walk, not a single-prim read, and stays registered.
REMOVED_ALIASES = {
    "get_isaac_prim_info",
    "get_isaac_prim_transform",
    "get_isaac_prim_ancestors",
    "get_isaac_prim_relationships",
    "get_isaac_prim_variants",
    "get_isaac_bounding_box",
    "get_isaac_mesh_info",
    "get_isaac_light_info",
    "get_isaac_material_info",
    "get_isaac_rigid_body_info",
    "get_isaac_collision_info",
    "get_isaac_joint_info",
    "get_isaac_mass_properties",
    "get_isaac_animation_info",
}


class _RecordingFastMCP:
    """FastMCP double that records registered tool names."""

    def __init__(self, name: str, version: str, **kwargs: Any):
        self.name = name
        self.version = version
        self.tools: List[SimpleNamespace] = []

    def tool(
        self, name: str, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(SimpleNamespace(name=name, func=func))
            return func

        return decorator

    def resource(self, *args: Any, **kwargs: Any):
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def add_middleware(self, middleware: Any) -> None:
        return


def _registered_tool_names(monkeypatch: pytest.MonkeyPatch) -> Set[str]:
    """All tool names the server registers with real backend availability."""
    monkeypatch.setattr(server_module, "FastMCP", _RecordingFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    instance = server_module.SimulMCPServer(settings=Settings())
    return {tool.name for tool in instance.mcp.tools}


def test_removed_prim_getter_aliases_stay_unregistered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = _registered_tool_names(monkeypatch)
    still_registered = REMOVED_ALIASES & names
    assert (
        not still_registered
    ), f"Removed alias tools re-registered: {sorted(still_registered)}"
    # The consolidation target and the deliberate survivor must both exist.
    assert "get_isaac_prim_detail" in names
    assert "get_isaac_texture_dependencies" in names


def test_every_skill_tool_reference_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = _registered_tool_names(monkeypatch)
    referenced: dict = {}
    for path in (_REPO / "skills").rglob("*.md"):
        for match in re.finditer(r"mcp__simul__([a-z_]+)", path.read_text()):
            referenced.setdefault(match.group(1), set()).add(
                str(path.relative_to(_REPO))
            )
    assert referenced, "skills/ tree has no tool references — glob broken?"
    dangling = {
        name: sorted(files) for name, files in referenced.items() if name not in names
    }
    assert not dangling, f"skills reference unregistered tools: {dangling}"
