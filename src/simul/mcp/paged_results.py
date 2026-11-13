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

A tool whose bulk is one mapping rather than a list (``get_prim_info``'s
``attributes``, keyed by name) declares that mapping in ``BULK_FIELDS``. The
budget then cuts it, to the entries that fit, before it looks at any list, so
the small lists beside it (the prim's applied API schemas) are not emptied
in its place.

The envelope names the tool whose result is being sent (``tool_result``);
the budget reads it with ``current_paging`` and ``current_bulk``.
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
        count_maps: Result fields mapping each paged list's name to its
            length (``returned: {"traversable_surfaces": 40, ...}``); restated
            to the lengths kept.
    """

    fields: Tuple[str, ...]
    offset_param: str = "offset"
    limit_param: Optional[str] = "max_results"
    count_fields: Tuple[str, ...] = ("count", "returned")
    reset_fields: Tuple[str, ...] = ()
    count_maps: Tuple[str, ...] = ()


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
    "get_unreal_physics_state": PagedList(("bodies",)),
    "list_unreal_simulation_actors": PagedList(("actors",)),
    # Both categories paged together at one offset; returned counts each.
    "analyze_unreal_scene_for_robotics": PagedList(
        ("traversable_surfaces", "graspable_objects"), count_maps=("returned",)
    ),
}


@dataclass(frozen=True)
class BulkField:
    """The mapping that carries a tool's result, cut first when the result is too large.

    Attributes:
        field: The top-level key of the mapping (``"attributes"``).
        hint: How to ask for the entries the cut leaves out, named in the
            cut's record.
    """

    field: str
    hint: str


#: Tools whose result is mostly one mapping, by registered tool name.
BULK_FIELDS: Dict[str, BulkField] = {
    "get_prim_info": BulkField(
        "attributes",
        "Pass attribute_filter (glob patterns such as physics:* or xformOp:*) to read the "
        "attributes left out, or a smaller array_preview to fit more of them.",
    ),
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


def current_tool() -> Optional[str]:
    """The tool named by the enclosing ``tool_result``, if any."""
    return _CURRENT_TOOL.get()


def current_paging() -> Optional[PagedList]:
    """The declaration for the tool named by the enclosing ``tool_result``."""
    return paging_for(_CURRENT_TOOL.get())


def bulk_for(tool_name: Optional[str]) -> Optional[BulkField]:
    """The bulk mapping ``tool_name`` declares, or None."""
    return BULK_FIELDS.get(tool_name) if tool_name else None


def current_bulk() -> Optional[BulkField]:
    """The bulk mapping of the tool named by the enclosing ``tool_result``."""
    return bulk_for(_CURRENT_TOOL.get())


__all__ = [
    "BULK_FIELDS",
    "BulkField",
    "PAGED_RESULTS",
    "PagedList",
    "bulk_for",
    "current_bulk",
    "current_paging",
    "current_tool",
    "paging_for",
    "tool_result",
]
