"""A size ceiling for tool results.

Tool results land directly in the calling agent's context window, and the only
guard in the path was the transport's 10 MB response cap — roughly 2.5M tokens,
larger than any context window. A single scene-inspection call against a
production stage could therefore crowd out the conversation it was meant to
inform.

The budget cuts what is actually costing the bytes and says so in the payload,
so the caller can narrow its query rather than wonder why the numbers look
short. Truncating keeps a usable prefix: a partial answer with an honest marker
beats both a silent cut and a bare refusal.

A paged result is cut as a page. Each paged tool declares which lists its
``offset`` pages (``paged_results.PAGED_RESULTS``); those lists are cut together
to the same number of entries, never fewer than one, and ``next_offset`` (with a
``count`` or ``returned`` that counted the page) is restated to continue right
after what was kept, so following ``next_offset`` neither skips entries nor
returns the same page forever. A page whose first entry alone is too large
sends that entry compacted to what identifies it; when that keeps every entry
of the page, ``next_offset`` stays as the tool set it (None on the last page).
Any other list the budget has to cut is reported as cut for good:
``next_offset`` is left as the tool set it, and the notice says the dropped
entries will not appear on later pages.

A tool whose bulk is one mapping declares it (``paged_results.BULK_FIELDS``,
``get_prim_info``'s ``attributes``). That mapping is cut first, to its
leading entries that fit, and named in ``truncated_fields`` with the tool's
hint, so a small list beside it is never emptied in its place.

A payload whose bulk is a string cannot be trimmed the same way — guessing where
to cut an opaque blob is more likely to corrupt it than to help — so up to a
second, hard ceiling it is passed through and flagged. Above the hard ceiling
the field carrying the excess is cut wherever it sits, nested fields included:
a string is replaced by a marker naming the field and its size, and a list or
mapping keeps the prefix that fits and is named in ``truncated_fields``. One
call must not be able to evict the conversation it was meant to inform.

Every result is also valid JSON. ``json.dumps`` writes a non-finite float as the
bare token ``Infinity`` or ``NaN``, which strict parsers reject, so such floats
become the strings ``"Infinity"``, ``"-Infinity"`` and ``"NaN"``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from ..logging import get_logger
from .paged_results import (
    BulkField,
    PagedList,
    bulk_for,
    current_bulk,
    current_paging,
    current_tool,
    paging_for,
    tool_result,
)

logger = get_logger(__name__)

# ~40 KB ≈ 10k tokens: large enough for a real listing, small enough that a
# single call cannot dominate a context window.
DEFAULT_RESULT_BUDGET_BYTES = 40_000

# ~400 KB ≈ 100k tokens. Nothing the caller can read is worth this much of its
# context; a field this large is cut rather than passed through.
HARD_RESULT_LIMIT_BYTES = 400_000

# For a cut that is not a declared page. It names no parameters: each tool
# narrows its results in its own way.
_DEFAULT_HINT = "Result truncated to fit the response budget; ask for less to see the rest."

_HARD_LIMIT_HINT = (
    "Field replaced: it exceeded the hard result limit. Write large output to "
    "a file and return its path, or return less of it."
)

_HARD_TRIM_HINT = (
    "Field cut to fit the hard result limit; the items after `returned` were "
    "dropped. Ask for less of it (a filter, a smaller preview or page)."
)

# A field smaller than this is never worth cutting on its own.
_MIN_CUT_BYTES = 256

# Bound on hard-limit passes; each one cuts the field carrying the excess.
_MAX_HARD_LIMIT_PASSES = 32

# What a compacted page entry keeps: these keys, and any value this small.
_IDENTITY_KEYS = frozenset({
    "path", "prim_path", "actor_path", "name", "object_name", "bone_name", "id", "label", "type",
})
_COMPACT_VALUE_BYTES = 256

PathKey = Union[str, int]


def _encode(payload: Any) -> str:
    return json.dumps(payload, default=str)


def _encoded_size(payload: Any) -> int:
    return len(_encode(payload))


def _json_float(value: float) -> Any:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _finite(value: Any) -> Any:
    """Return ``value`` with every non-finite float replaced by its string spelling."""
    if isinstance(value, float):
        return _json_float(value)
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _encode_strict(payload: Any) -> Tuple[Any, str]:
    """Encode ``payload`` as valid JSON; return the (possibly cleaned) payload too.

    The common case encodes once. Only a payload holding ``inf`` or ``nan``
    is walked and encoded again.
    """
    try:
        return payload, json.dumps(payload, default=str, allow_nan=False)
    except ValueError:
        cleaned = _finite(payload)
        return cleaned, _encode(cleaned)


def _list_size(item_sizes: List[int], count: int) -> int:
    """Encoded length of a list's first ``count`` items, from per-item sizes.

    ``json.dumps`` without ``indent`` writes a list as ``[`` + items joined by
    ``", "`` + ``]`` and encodes each item independently of its neighbours, so
    the prefix length follows from the item lengths without re-encoding.
    """
    if count <= 0:
        return 2
    return 2 + sum(item_sizes[:count]) + 2 * (count - 1)


def _largest_list_field(
    payload: Dict[str, Any], exclude: frozenset = frozenset()
) -> Optional[Tuple[str, List[Any], List[int]]]:
    """Return (key, list, per-item sizes) for the top-level list costing the most bytes.

    Keys in ``exclude`` (the declared paged lists, already cut) are skipped.
    """
    best: Optional[Tuple[str, List[Any], List[int]]] = None
    best_size = -1
    for key, value in payload.items():
        if key in exclude or not (isinstance(value, list) and value):
            continue
        item_sizes = [_encoded_size(item) for item in value]
        size = _list_size(item_sizes, len(item_sizes))
        # Strictly greater keeps the first of equal-sized lists, as max() does.
        if size > best_size:
            best, best_size = (key, value, item_sizes), size
    return best


class _FromContext:
    """Sentinel: take the paging declaration from the enclosing ``tool_result``."""


_FROM_CONTEXT = _FromContext()


def _resolve_paging(tool_name: Optional[str], paging: Any) -> Optional[PagedList]:
    if paging is not _FROM_CONTEXT:
        return paging
    if tool_name is not None:
        return paging_for(tool_name)
    return current_paging()


def _resolve_bulk(tool_name: Optional[str]) -> Optional[BulkField]:
    return bulk_for(tool_name) if tool_name is not None else current_bulk()


def apply_result_budget(
    payload: Any,
    budget_bytes: int = DEFAULT_RESULT_BUDGET_BYTES,
    hint: str = _DEFAULT_HINT,
    hard_limit_bytes: int = HARD_RESULT_LIMIT_BYTES,
    *,
    tool_name: Optional[str] = None,
    paging: Any = _FROM_CONTEXT,
) -> Any:
    """Trim ``payload`` so its JSON encoding fits ``budget_bytes``.

    A paged tool's page (its ``PagedList`` declaration) is cut first: every
    declared list to the same number of entries, at least one, and the paging
    fields restated so ``next_offset`` continues right after what was kept.
    Then, if still over budget, the largest other top-level list is cut and
    reported as a cut later pages will not repeat, with ``next_offset`` left
    as the tool set it. A payload whose bulk is elsewhere is flagged with
    ``oversized_bytes`` and passed through until it exceeds
    ``hard_limit_bytes``; then the fields carrying the excess are cut at any
    depth (a string becomes a ``truncated_field`` marker, a list or mapping
    keeps its prefix and is listed in ``truncated_fields``). Non-finite floats
    become strings.

    Args:
        payload: The tool's result; anything but a dict is returned unchanged
            apart from non-finite floats.
        budget_bytes: Soft ceiling met by trimming lists.
        hint: Text placed in the notice of a cut that is not a page.
        hard_limit_bytes: Ceiling above which fields are cut at any depth.
        tool_name: The tool whose result this is, to look up its paging.
        paging: The paging declaration itself (None for none); by default the
            one for ``tool_name``, or for the tool the envelope names
            (``paged_results.tool_result``).

    Returns:
        The payload unchanged when it already fits, else the trimmed copy.
    """
    return _budget(
        payload, budget_bytes, hint, hard_limit_bytes, _resolve_paging(tool_name, paging), _resolve_bulk(tool_name)
    )[0]


def encode_result_budget(
    payload: Any,
    budget_bytes: int = DEFAULT_RESULT_BUDGET_BYTES,
    hint: str = _DEFAULT_HINT,
    hard_limit_bytes: int = HARD_RESULT_LIMIT_BYTES,
    *,
    tool_name: Optional[str] = None,
    paging: Any = _FROM_CONTEXT,
) -> str:
    """Return ``json.dumps(apply_result_budget(payload, ...), default=str)``.

    Byte-identical to that expression, but the encoding made to measure the
    payload is the one returned, so a payload that already fits is encoded
    once instead of twice.
    """
    return _budget(
        payload, budget_bytes, hint, hard_limit_bytes, _resolve_paging(tool_name, paging), _resolve_bulk(tool_name)
    )[1]


def _budget(
    payload: Any,
    budget_bytes: int,
    hint: str,
    hard_limit_bytes: int,
    paging: Optional[PagedList],
    bulk: Optional[BulkField] = None,
) -> Tuple[Any, str]:
    """Apply the budget; return the resulting payload and its JSON encoding."""
    payload, text = _encode_strict(payload)
    if not isinstance(payload, dict) or len(text) <= budget_bytes:
        return payload, text
    paged_roots: frozenset = frozenset()
    if paging is not None:
        lists = _paged_lists(payload, paging)
        if lists:
            paged_roots = frozenset(str(path[0]) for path, _items in lists)
            # The page keeps up to half the budget (all of it when it is
            # smaller); other content larger than the rest is cut first, as a
            # cut later pages will not repeat.
            rest = _encoded_size(_emptied(payload, lists))
            page_bytes = len(text) - rest
            reserve = min(page_bytes, budget_bytes // 2)
            if rest > budget_bytes - reserve:
                payload, text = _trim_largest_list(
                    payload, len(text), budget_bytes - reserve + page_bytes, hint, paged_roots, flag=False
                )
                lists = _paged_lists(payload, paging)
            payload, text = _trim_page(payload, len(text), budget_bytes, paging, lists)
    if len(text) > budget_bytes and bulk is not None:
        payload, text = _trim_bulk(payload, len(text), budget_bytes, bulk, paged_roots)
    if len(text) > budget_bytes:
        payload, text = _trim_largest_list(payload, len(text), budget_bytes, hint, paged_roots)
    return _enforce_hard_limit(payload, text, hard_limit_bytes)


# ---------------------------------------------------------------------------
# The declared bulk mapping
# ---------------------------------------------------------------------------


def _trim_bulk(
    payload: Dict[str, Any], payload_size: int, budget_bytes: int, bulk: BulkField, paged_roots: frozenset
) -> Tuple[Dict[str, Any], str]:
    """Cut the declared bulk mapping to its leading entries that fit the budget.

    Only when the mapping outweighs every top-level list: then it is what
    makes the payload too large, and cutting it keeps the small fields and
    lists beside it whole. Otherwise the largest list is cut first, as for any
    tool. The cut is recorded in ``truncated_fields`` with the tool's hint;
    ``next_offset`` (if any) is not touched.
    """
    mapping = payload.get(bulk.field)
    if not isinstance(mapping, dict) or not mapping:
        return payload, _encode(payload)
    mapping_size = _encoded_size(mapping)
    largest = _largest_list_field(payload, paged_roots)
    if largest is not None and _list_size(largest[2], len(largest[2])) > mapping_size:
        return payload, _encode(payload)
    items = list(mapping.items())
    total = len(items)
    record: Dict[str, Any] = {"field": bulk.field, "returned": total, "total": total, "hint": bulk.hint}
    if "next_offset" in payload:
        record["next_offset"] = "unchanged"
    first_left = {"first_omitted": max((key for key, _value in items), key=len)}
    notes = {"truncated": True, "truncated_fields": [*payload.get("truncated_fields", []), {**record, **first_left}]}
    remaining = budget_bytes - _encoded_size({**payload, bulk.field: {}}) - _encoded_size(notes)
    # A mapping encodes as "{" + entries joined by ", " + "}", each entry
    # '"key": value', measured once.
    entry_sizes = [len(json.dumps(key)) + 2 + _encoded_size(value) for key, value in items]
    kept, used = 0, 2
    for size in entry_sizes:
        grown = used + size + (2 if kept else 0)
        if grown > remaining:
            break
        kept, used = kept + 1, grown
    if kept >= total:
        return payload, _encode(payload)
    trimmed = dict(payload)
    trimmed[bulk.field] = dict(items[:kept])
    trimmed["truncated"] = True
    record.update(returned=kept, first_omitted=items[kept][0])
    trimmed["truncated_fields"] = [*payload.get("truncated_fields", []), record]
    return trimmed, _encode(trimmed)


def _emptied(payload: Dict[str, Any], lists: List[Tuple[Tuple[PathKey, ...], List[Any]]]) -> Any:
    """``payload`` with every paged list replaced by an empty one."""
    emptied: Any = payload
    for path, _items in lists:
        emptied = _replace_at(emptied, path, [])
    return emptied


# ---------------------------------------------------------------------------
# The declared page
# ---------------------------------------------------------------------------


def _pattern(field: str) -> List[Union[str, None]]:
    """Split ``"a[*].b"`` into ``["a", None, "b"]``; None stands for every list entry."""
    steps: List[Union[str, None]] = []
    for part in field.split("."):
        wildcard = part.endswith("[*]")
        steps.append(part[:-3] if wildcard else part)
        if wildcard:
            steps.append(None)
    return steps


def _paged_lists(payload: Dict[str, Any], paging: PagedList) -> List[Tuple[Tuple[PathKey, ...], List[Any]]]:
    """The concrete lists the declaration names in this payload, with their paths."""
    found: List[Tuple[Tuple[PathKey, ...], List[Any]]] = []

    def walk(node: Any, steps: List[Union[str, None]], path: Tuple[PathKey, ...]) -> None:
        if not steps:
            if isinstance(node, list):
                found.append((path, node))
            return
        step, rest = steps[0], steps[1:]
        if step is None:
            if isinstance(node, list):
                for index, child in enumerate(node):
                    walk(child, rest, path + (index,))
        elif isinstance(node, dict) and step in node:
            walk(node[step], rest, path + (step,))

    for field in paging.fields:
        walk(payload, _pattern(field), ())
    return found


def _compact_item(item: Any) -> Any:
    """Shrink one oversized page entry to what identifies it.

    A mapping keeps its identifying keys and its small values and says which
    keys it dropped; anything else becomes a marker. The entry still occupies
    its position, so the caller can page past it and fetch it on its own.
    """
    original = _encoded_size(item)
    if not isinstance(item, dict):
        return {"truncated_item": True, "original_bytes": original, "value_type": type(item).__name__}
    kept: Dict[str, Any] = {}
    dropped: List[str] = []
    for key, value in item.items():
        if key in _IDENTITY_KEYS or _encoded_size(value) <= _COMPACT_VALUE_BYTES:
            kept[key] = value
        else:
            dropped.append(key)
    kept["truncated_item"] = True
    kept["dropped_keys"] = dropped
    kept["original_bytes"] = original
    return kept


def _trim_page(
    payload: Dict[str, Any],
    payload_size: int,
    budget_bytes: int,
    paging: PagedList,
    lists: List[Tuple[Tuple[PathKey, ...], List[Any]]],
) -> Tuple[Dict[str, Any], str]:
    """Cut every declared list to the same entry count and restate the paging.

    The count is the largest that fits the budget, and never zero: a page
    whose first entry alone is too large keeps that entry, compacted, so
    ``next_offset`` always moves past it. When the paged lists are not what
    makes the payload too large, the payload is returned unchanged.
    """
    longest = max(len(items) for _path, items in lists)
    if longest == 0 or payload_size <= budget_bytes:
        return payload, _encode(payload)
    item_sizes = [[_encoded_size(item) for item in items] for _path, items in lists]
    emptied = _emptied(payload, lists)
    offset = _page_offset(payload, longest)
    new_next = offset + longest
    notice = _page_notice(paging, longest, longest, new_next)
    overhead = _encoded_size(emptied) + _encoded_size(
        {"truncated": True, "next_offset": new_next, "truncation": notice}
    )

    def size_at(count: int) -> int:
        return overhead + sum(_list_size(sizes, min(count, len(sizes))) - 2 for sizes in item_sizes)

    if size_at(longest) <= budget_bytes:
        # The page is not what is too large; the other content is cut later.
        return payload, _encode(payload)
    low, high = 0, longest
    while low < high:
        middle = (low + high + 1) // 2
        if size_at(middle) <= budget_bytes:
            low = middle
        else:
            high = middle - 1
    # Never zero: a page that keeps nothing would send next_offset == offset,
    # and a caller following it would get this page forever.
    kept = max(1, low)
    pages = [items[:kept] for _path, items in lists]
    compacted: List[str] = []
    if low == 0:
        # Even one entry per list is too large: compact first entries, largest
        # first, until the page fits; each still holds its position.
        current = size_at(1)
        order = sorted(range(len(lists)), key=lambda i: item_sizes[i][0] if item_sizes[i] else 0, reverse=True)
        for index in order:
            if current <= budget_bytes:
                break
            if not pages[index]:
                continue
            small = _compact_item(pages[index][0])
            current -= item_sizes[index][0] - _encoded_size(small)
            pages[index] = [small]
            compacted.append(_dotted(lists[index][0] + (0,)))

    trimmed: Any = payload
    for (path, _items), page in zip(lists, pages):
        trimmed = _replace_at(trimmed, path, page)

    old_next = payload.get("next_offset")
    # Every entry kept (one compacted): the page still ends where the tool
    # said, so next_offset stays as it was, None on the last page.
    whole = kept == longest
    new_next = old_next if whole else offset + kept
    trimmed["next_offset"] = new_next
    trimmed["truncated"] = True
    for key in paging.count_fields:
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value == longest:
            trimmed[key] = kept
    for key in paging.reset_fields:
        if key in trimmed:
            trimmed[key] = None
    for key in paging.count_maps:
        counts = payload.get(key)
        if isinstance(counts, dict):
            restated = dict(counts)
            for path, _items in lists:
                if len(path) == 1 and path[0] in restated:
                    restated[path[0]] = len(trimmed[path[0]])
            trimmed[key] = restated
    stale = payload.get("hint")
    if not whole and isinstance(old_next, int) and isinstance(stale, str):
        # "offset=1000" (Unreal's listings) or "offset 1000" (its robotics page).
        trimmed["hint"] = re.sub(rf"(\boffset[ =]){old_next}\b", rf"\g<1>{new_next}", stale)
    trimmed["truncation"] = _page_notice(paging, kept, longest, new_next, moved=not whole)
    if compacted:
        trimmed["truncation"]["compacted_entries"] = compacted
        trimmed["truncation"]["hint"] += (
            f" {', '.join(compacted)} did not fit on its own and is sent compacted (its identifying "
            "and small fields; dropped_keys lists the rest): ask for less of each entry to read it whole."
        )
    return trimmed, _encode(trimmed)


def _page_offset(payload: Dict[str, Any], page_length: int) -> int:
    """Where this page starts: ``offset`` as sent, else inferred from ``next_offset``."""
    offset = payload.get("offset")
    if isinstance(offset, int) and not isinstance(offset, bool):
        return offset
    following = payload.get("next_offset")
    if isinstance(following, int) and not isinstance(following, bool) and following >= page_length:
        return following - page_length
    return 0


def _page_notice(
    paging: PagedList, kept: int, total: int, next_offset: Optional[int], moved: bool = True
) -> Dict[str, Any]:
    if not moved:
        following = (
            "this is the last page" if next_offset is None
            else f"call again with {paging.offset_param}={next_offset} for the next page"
        )
        summary = (
            f"All {kept} entries kept, compacted to fit the response budget; next_offset is "
            f"unchanged: {following}."
        )
    else:
        advice = f"call again with {paging.offset_param}={next_offset}"
        if paging.limit_param:
            advice += f", or pass a smaller {paging.limit_param}"
        summary = (
            f"Page cut to {kept} of {total} entries to fit the response budget; "
            f"next_offset moved to {next_offset}: {advice}."
        )
    return {
        "field": paging.fields[0],
        "fields": list(paging.fields),
        "returned": kept,
        "total": total,
        "next_offset": next_offset,
        "hint": summary,
    }


# ---------------------------------------------------------------------------
# Lists that are not the page
# ---------------------------------------------------------------------------


def _trim_largest_list(
    payload: Dict[str, Any],
    payload_size: int,
    budget_bytes: int,
    hint: str,
    paged_roots: frozenset = frozenset(),
    flag: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """Cut the largest top-level list that is not the page, or flag the payload oversized.

    ``payload_size`` is the payload's already-measured encoded length. The
    cut never touches ``next_offset``: the list is not what the offset pages,
    so its dropped entries do not come back on a later page, and the notice
    says so. Returns the resulting payload together with its encoding; with
    ``flag`` False, a payload it cannot usefully trim comes back unchanged
    instead of flagged ``oversized_bytes``.
    """

    def _oversized() -> Tuple[Dict[str, Any], str]:
        """Report an over-budget payload we could not usefully trim."""
        if not flag:
            return payload, _encode(payload)
        noted = dict(payload)
        noted["oversized_bytes"] = payload_size
        return noted, _encode(noted)

    largest = _largest_list_field(payload, paged_roots)
    if largest is None:
        return _oversized()

    field, items, item_sizes = largest
    total = len(items)
    paged = "next_offset" in payload or bool(paged_roots)
    if "next_offset" in payload and not paged_roots:
        # A paged result with no declaration in paged_results.PAGED_RESULTS:
        # its page is cut as if it were not one. Say so where it can be fixed.
        logger.warning(
            "Result budget cut %r of an undeclared paged result%s; declare its page in "
            "simul.mcp.paged_results.PAGED_RESULTS",
            field,
            f" from {current_tool()}" if current_tool() else "",
        )
    if paged:
        hint = (
            f"{field} was cut to fit the response budget. It is not the list next_offset pages, so "
            "its dropped entries will not appear on later pages; next_offset is unchanged. Ask for "
            "less (a filter or a smaller page) to see them."
        )
    notice: Dict[str, Any] = {"field": field, "returned": total, "total": total, "hint": hint}
    if paged:
        notice["next_offset"] = "unchanged"
    key = "truncated_fields" if paged_roots else "truncation"

    def notes(entry: Dict[str, Any]) -> Dict[str, Any]:
        if key == "truncation":
            return {"truncation": entry}
        return {"truncated_fields": [*payload.get("truncated_fields", []), entry]}

    # Size of everything except the list, plus room for the truncation notice.
    overhead = _encoded_size({**payload, field: []}) + _encoded_size({"truncated": True, **notes(notice)})
    remaining = budget_bytes - overhead
    if remaining <= 0:
        kept = 0
    else:
        # Estimate from the average item, then shrink until it genuinely fits.
        # Prefix sizes come from the per-item sizes measured once above.
        average = max(1, _list_size(item_sizes, total) // total)
        kept = max(0, min(total, remaining // average))
        while kept > 0 and _list_size(item_sizes, kept) > remaining:
            kept = int(kept * 0.9) if kept > 10 else kept - 1

    trimmed = dict(payload)
    trimmed[field] = items[:kept]
    trimmed["truncated"] = True
    trimmed.update(notes({**notice, "returned": kept}))
    trimmed_text = _encode(trimmed)

    # Only keep the trim if it actually helped. The largest *top-level* list is
    # not always where the bulk is: get_isaac_prim_detail carries its weight in
    # dict values with one small "aspects" list, and prim info carries it in
    # "attributes" while the only list is "children". Trimming those drops
    # metadata the caller needs, reports a truncation that did not happen, and
    # leaves the payload over budget — larger, in fact, since the notice costs
    # more than the list did.
    if len(trimmed_text) < payload_size:
        return trimmed, trimmed_text

    # Nothing worth trimming at the top level. Say the payload is oversized
    # rather than pretending it was cut; a caller acting on a false
    # "truncated" flag would go looking for data that is all still here.
    return _oversized()


# ---------------------------------------------------------------------------
# The hard limit
# ---------------------------------------------------------------------------


def _dotted(path: Tuple[PathKey, ...]) -> str:
    """Name a nested field: ``attributes.points.value``, ``results[3].name``."""
    text = ""
    for key in path:
        if isinstance(key, int):
            text += f"[{key}]"
        else:
            text += f".{key}" if text else str(key)
    return text


def _child_items(node: Any) -> List[Tuple[PathKey, Any]]:
    if isinstance(node, dict):
        return list(node.items())
    if isinstance(node, (list, tuple)):
        return list(enumerate(node))
    return []


def _largest_child(
    node: Any, node_size: int, excess: int, nested: bool
) -> Optional[Tuple[PathKey, Any, int]]:
    """Return (key, child, encoded size) of ``node``'s largest container or string child.

    Below the top level the scan stops once the bytes not yet measured could
    no longer hold a child that carries the excess or beats the largest so
    far, so a long list of small records is not encoded item by item. Scalars
    are skipped: they are never worth cutting alone.
    """
    best: Optional[Tuple[PathKey, Any, int]] = None
    measured = 0
    for key, child in _child_items(node):
        if not isinstance(child, (dict, list, tuple, str)):
            continue
        size = _encoded_size(child)
        measured += size
        if best is None or size > best[2]:
            best = (key, child, size)
        if nested and node_size - measured < max(excess, best[2]):
            break
    return best


def _locate_excess(
    payload: Dict[str, Any], payload_size: int, excess: int
) -> Optional[Tuple[Tuple[PathKey, ...], Any, int]]:
    """Find the field to cut so the payload sheds ``excess`` bytes.

    Descends from the top into the largest child for as long as that child
    alone carries the excess, so the cut lands on the innermost field that
    explains the size (``attributes.points.value``, not ``attributes``). At the
    top level, where no single field carries it all, the largest field is cut
    whole and the caller measures again.

    Returns:
        (path, node, encoded size) of the field to cut, or None when nothing
        is large enough to be worth cutting.
    """
    path: Tuple[PathKey, ...] = ()
    node: Any = payload
    node_size = payload_size
    while True:
        largest = _largest_child(node, node_size, excess, nested=bool(path))
        if largest is None:
            break
        key, child, size = largest
        if size < _MIN_CUT_BYTES or (size < excess and path):
            # Nothing below is worth cutting alone: cut this container.
            break
        path, node, node_size = path + (key,), child, size
        if size < excess or isinstance(child, str):
            # A top-level field that cannot shed it all, or a string: cut it whole.
            return path, node, size
    if not path:
        return None
    return path, node, node_size


def _prefix_fitting(node: Any, size: int, target: int) -> Tuple[Any, int]:
    """Return the longest prefix of a list or mapping that encodes within ``target`` bytes.

    ``size`` is ``node``'s encoded size; the first guess scales it, and the
    guess shrinks until the prefix really fits.
    """
    is_mapping = isinstance(node, dict)
    items = list(node.items()) if is_mapping else list(node)

    def build(count: int) -> Any:
        return dict(items[:count]) if is_mapping else items[:count]

    total = len(items)
    if total == 0 or target <= 2:
        return build(0), 0
    kept = min(total, max(0, int(total * (target - 2) / max(1, size - 2))))
    while kept > 0 and _encoded_size(build(kept)) > target:
        kept = int(kept * 0.9) if kept > 10 else kept - 1
    return build(kept), kept


def _replace_at(root: Any, path: Tuple[PathKey, ...], value: Any) -> Any:
    """Return a copy of ``root`` with the node at ``path`` replaced; ``root`` is unchanged."""
    if not path:
        return value
    head, rest = path[0], path[1:]
    copy: Any = dict(root) if isinstance(root, dict) else list(root)
    copy[head] = _replace_at(root[head], rest, value)
    return copy


def _enforce_hard_limit(
    payload: Dict[str, Any], text: str, hard_limit_bytes: int
) -> Tuple[Dict[str, Any], str]:
    """Cut the fields carrying the excess, at any depth, until the payload fits.

    A string is replaced by a ``truncated_field`` marker. A list or mapping
    keeps the prefix that fits and is recorded in ``truncated_fields``. Paging
    is never restated here: by now a declared page has been cut to the budget,
    so what this cuts is other content, and a paged payload's record says
    ``next_offset`` is unchanged.

    ``text`` is ``payload``'s current encoding; the pair returned stays in step.
    """
    for _ in range(_MAX_HARD_LIMIT_PASSES):
        if len(text) <= hard_limit_bytes:
            break
        excess = len(text) - hard_limit_bytes
        located = _locate_excess(payload, len(text), excess)
        if located is None:
            break
        path, node, size = located
        field = _dotted(path)
        record: Optional[Dict[str, Any]] = None
        if isinstance(node, str):
            replacement: Any = {
                "truncated_field": field,
                "original_bytes": len(node.encode("utf-8")),
                "hint": _HARD_LIMIT_HINT,
            }
            if _encoded_size(replacement) >= size:
                # The remaining strings are all small; replacing them cannot help.
                break
        else:
            total = len(node)
            replacement, kept = _prefix_fitting(node, size, max(2, size - excess - 512))
            if kept >= total:
                break
            record = {"field": field, "returned": kept, "total": total, "hint": _HARD_TRIM_HINT}
            if "next_offset" in payload:
                record["next_offset"] = "unchanged"
        cut = _replace_at(payload, path, replacement)
        if record is not None:
            cut["truncated"] = True
            cut["truncated_fields"] = [*payload.get("truncated_fields", []), record]
        cut_text = _encode(cut)
        if len(cut_text) >= len(text):
            break
        payload, text = cut, cut_text
    return payload, text


__all__ = [
    "DEFAULT_RESULT_BUDGET_BYTES",
    "HARD_RESULT_LIMIT_BYTES",
    "PagedList",
    "apply_result_budget",
    "encode_result_budget",
    "tool_result",
]
