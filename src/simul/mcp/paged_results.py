"""Which list in a tool's result is its page, so the result budget can page it honestly.

A paged tool returns one page of a longer listing and a ``next_offset`` where
the next page starts. When the result budget (``result_budget``) has to cut
such a page to fit, it must cut the paged list and move ``next_offset`` back
to just past what it kept; otherwise a caller following ``next_offset`` skips
everything the cut removed. Guessing which list that is went wrong twice: the
budget trimmed the largest list, which for ``list_isaac_physics_objects`` is
one of three lists sharing an offset, and for a Blender trajectory page is the
list of targets while the offset counts samples.

So each paged tool declares its paged lists here, by tool name, and the budget
rewrites paging fields only for what is declared. A result from a tool with no
declaration is still cut to fit, but its ``next_offset`` is left alone and the
cut is reported as one later pages will not repeat.

The envelope names the tool whose result is being sent (``tool_result``);
the budget reads it with ``current_paging``.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple


@dataclass(frozen=True)
class PagedList:
    """How one tool pages its result.

    Attributes:
        fields: Paths of the lists one offset pages, all advanced together:
            ``"prims"``; ``"root.children"`` for a list inside a mapping;
            ``"trajectories[*].points"`` for a list inside every entry of
            another list. ``offset`` and ``next_offset`` count entries of each
            of them, so entry ``i`` of every list belongs to position
            ``offset + i``.
        offset_param: The tool parameter that takes ``next_offset``.
        limit_param: The tool parameter that sets the page size, named in the
            hint when a page is cut; None when the tool has none.
        count_fields: Result fields that count the entries of the page; one
            equal to the page's length is restated to the entries kept.
        reset_fields: Secondary cursors into the page's last entry (Blender's
            ``next_key_offset``); a cut page ends on another entry, so they
            are cleared to None.
    """

    fields: Tuple[str, ...]
    offset_param: str = "offset"
    limit_param: Optional[str] = "max_results"
    count_fields: Tuple[str, ...] = ("count", "returned")
    reset_fields: Tuple[str, ...] = ()


#: Every paged tool's declaration, by registered tool name.
#: ``tests/mcp/test_paged_results.py`` checks each against the registered
#: tool (its parameters and its result fields) and fails for a paged tool
#: that is missing here.
PAGED_RESULTS: Dict[str, PagedList] = {
    # Isaac Sim
    "list_isaac_prims": PagedList(("prims",)),
    "search_isaac_prims": PagedList(("matches",)),
    "get_isaac_subtree": PagedList(("prims",)),
    "query_isaac_typed_prims": PagedList(("prims",)),
    "list_isaac_lights": PagedList(("lights",)),
    "list_isaac_cameras": PagedList(("cameras",)),
    "list_isaac_materials": PagedList(("materials",)),
    "list_isaac_extensions": PagedList(("extensions",), limit_param="limit"),
    # Three lists sliced at one offset: entry i of each is position offset + i.
    "list_isaac_physics_objects": PagedList(("rigid_bodies", "colliders", "joints")),
    # Headless USD
    "search_prims": PagedList(("results",), limit_param="limit"),
    "get_prim_subtree": PagedList(("prims",)),
    # Blender
    "list_blender_scene_objects": PagedList(("objects",), limit_param="max_items"),
    "search_blender_objects": PagedList(("objects",)),
    "validate_simready_compliance": PagedList(("issues",), limit_param="limit"),
    # Channels from offset; key_offset pages the keys of the first channel.
    "get_blender_keyframes": PagedList(
        ("channels",), limit_param="limit", reset_fields=("next_key_offset",)
    ),
    # The offset counts trajectory points (samples), not targets: every
    # target's points are one page.
    "get_blender_object_trajectory": PagedList(("points",), limit_param="limit", count_fields=("point_count",)),
    "get_blender_trajectories": PagedList(
        ("trajectories[*].points",), limit_param="limit", count_fields=("point_count",)
    ),
    # Unreal Engine
    "list_unreal_actors": PagedList(("actors",)),
    "list_unreal_actor_components": PagedList(("components",)),
    "get_unreal_actor_by_semantic_label": PagedList(("actors",)),
    "query_unreal_scene_graph": PagedList(("root.children",)),
    "search_unreal_assets": PagedList(("assets",)),
}

_CURRENT_TOOL: ContextVar[Optional[str]] = ContextVar("simul_result_tool", default=None)


@contextmanager
def tool_result(tool_name: Optional[str]) -> Iterator[None]:
    """Name the tool whose result is encoded inside the block."""
    token = _CURRENT_TOOL.set(tool_name)
    try:
        yield
    finally:
        _CURRENT_TOOL.reset(token)


def paging_for(tool_name: Optional[str]) -> Optional[PagedList]:
    """The declaration for ``tool_name``, or None for a tool that does not page."""
    return PAGED_RESULTS.get(tool_name) if tool_name else None


def current_paging() -> Optional[PagedList]:
    """The declaration for the tool named by the enclosing ``tool_result``."""
    return paging_for(_CURRENT_TOOL.get())


__all__ = ["PAGED_RESULTS", "PagedList", "current_paging", "paging_for", "tool_result"]
