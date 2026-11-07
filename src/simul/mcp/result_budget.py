"""A size ceiling for tool results.

Tool results land directly in the calling agent's context window, and the only
guard in the path was the transport's 10 MB response cap — roughly 2.5M tokens,
larger than any context window. A single scene-inspection call against a
production stage could therefore crowd out the conversation it was meant to
inform.

The budget trims the field that is actually costing the bytes — the largest list
— and says so in the payload, so the caller can narrow its query rather than
wonder why the numbers look short. Truncating keeps a usable prefix: a partial
answer with an honest marker beats both a silent cut and a bare refusal.

When the trimmed list is a page of a paged listing (the payload carries
``offset`` or ``next_offset``), the paging fields are rewritten to match what
was kept: ``next_offset`` points just past the last item returned, and a
``count`` or ``returned`` that counted the page counts the kept items. A client
that follows ``next_offset`` then gets the items the trim removed instead of
skipping them.

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
from typing import Any, Dict, List, Optional, Tuple, Union

# ~40 KB ≈ 10k tokens: large enough for a real listing, small enough that a
# single call cannot dominate a context window.
DEFAULT_RESULT_BUDGET_BYTES = 40_000

# ~400 KB ≈ 100k tokens. Nothing the caller can read is worth this much of its
# context; a field this large is cut rather than passed through.
HARD_RESULT_LIMIT_BYTES = 400_000

_DEFAULT_HINT = (
    "Result truncated to fit the response budget. A paged result continues at "
    "next_offset; otherwise narrow the query (a more specific root_path, a "
    "filter, or a smaller page size) to see the rest."
)

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
    payload: Dict[str, Any],
) -> Optional[Tuple[str, List[Any], List[int]]]:
    """Return (key, list, per-item sizes) for the list contributing the most encoded bytes."""
    best: Optional[Tuple[str, List[Any], List[int]]] = None
    best_size = -1
    for key, value in payload.items():
        if not (isinstance(value, list) and value):
            continue
        item_sizes = [_encoded_size(item) for item in value]
        size = _list_size(item_sizes, len(item_sizes))
        # Strictly greater keeps the first of equal-sized lists, as max() does.
        if size > best_size:
            best, best_size = (key, value, item_sizes), size
    return best


def apply_result_budget(
    payload: Any,
    budget_bytes: int = DEFAULT_RESULT_BUDGET_BYTES,
    hint: str = _DEFAULT_HINT,
    hard_limit_bytes: int = HARD_RESULT_LIMIT_BYTES,
) -> Any:
    """Trim ``payload`` so its JSON encoding fits ``budget_bytes``.

    Only top-level list fields are trimmed to meet the budget; a trimmed page
    gets its paging fields restated. A payload whose bulk is elsewhere is
    flagged with ``oversized_bytes`` and passed through — until it exceeds
    ``hard_limit_bytes``. Then the fields carrying the excess are cut at any
    depth until the payload fits: a string becomes
    ``{"truncated_field": name, "original_bytes": n, "hint": ...}``, and a
    list or mapping keeps the prefix that fits and is listed in
    ``truncated_fields``. Non-finite floats become strings.

    Args:
        payload: The tool's result; anything but a dict is returned unchanged
            apart from non-finite floats.
        budget_bytes: Soft ceiling met by trimming lists.
        hint: Text placed in the truncation notice.
        hard_limit_bytes: Ceiling above which fields are cut at any depth.

    Returns:
        The payload unchanged when it already fits, else the trimmed copy.
    """
    return _budget(payload, budget_bytes, hint, hard_limit_bytes)[0]


def encode_result_budget(
    payload: Any,
    budget_bytes: int = DEFAULT_RESULT_BUDGET_BYTES,
    hint: str = _DEFAULT_HINT,
    hard_limit_bytes: int = HARD_RESULT_LIMIT_BYTES,
) -> str:
    """Return ``json.dumps(apply_result_budget(payload, ...), default=str)``.

    Byte-identical to that expression, but the encoding made to measure the
    payload is the one returned, so a payload that already fits is encoded
    once instead of twice.
    """
    return _budget(payload, budget_bytes, hint, hard_limit_bytes)[1]


def _budget(
    payload: Any, budget_bytes: int, hint: str, hard_limit_bytes: int
) -> Tuple[Any, str]:
    """Apply the budget; return the resulting payload and its JSON encoding."""
    payload, text = _encode_strict(payload)
    if not isinstance(payload, dict) or len(text) <= budget_bytes:
        return payload, text
    trimmed, trimmed_text = _trim_largest_list(payload, len(text), budget_bytes, hint)
    return _enforce_hard_limit(trimmed, trimmed_text, hard_limit_bytes)


def _trim_largest_list(
    payload: Dict[str, Any], payload_size: int, budget_bytes: int, hint: str
) -> Tuple[Dict[str, Any], str]:
    """Cut the largest top-level list until the payload fits, or flag it oversized.

    ``payload_size`` is the payload's already-measured encoded length. Returns
    the resulting payload together with its encoding.
    """

    def _oversized() -> Tuple[Dict[str, Any], str]:
        """Report an over-budget payload we could not usefully trim."""
        noted = dict(payload)
        noted["oversized_bytes"] = payload_size
        return noted, _encode(noted)

    largest = _largest_list_field(payload)
    if largest is None:
        return _oversized()

    field, items, item_sizes = largest
    total = len(items)

    # Size of everything except the list, plus room for the truncation notice.
    overhead = _encoded_size({**payload, field: []}) + _encoded_size(
        {"truncated": True, "truncation": {
            "field": field, "returned": total, "total": total, "hint": hint,
        }}
    )
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
    trimmed["truncation"] = {
        "field": field,
        "returned": kept,
        "total": total,
        "hint": hint,
    }
    _restate_paging(trimmed, payload, kept, total)
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


def _restate_paging(
    trimmed: Dict[str, Any], original: Dict[str, Any], kept: int, total: int
) -> None:
    """Make a paged payload's paging fields describe the page that was kept.

    A paged listing says where the next page starts (``next_offset``) and
    often how many items this page holds (``count`` or ``returned``). Cutting
    the page's list without updating them sends a client that follows
    ``next_offset`` past the items that were cut. ``count`` and ``returned``
    are rewritten only when they counted the list itself; a ``count`` that is
    a grand total stays as it is.

    Args:
        trimmed: The payload being trimmed, updated in place.
        original: The payload before the cut.
        kept: Items kept in the trimmed list.
        total: Items the list held before the cut.
    """
    offset = original.get("offset")
    has_offset = isinstance(offset, int) and not isinstance(offset, bool)
    if has_offset or "next_offset" in original:
        trimmed["next_offset"] = (offset if has_offset else 0) + kept
    for key in ("count", "returned"):
        value = original.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value == total:
            trimmed[key] = kept


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
    keeps the prefix that fits and is recorded in ``truncated_fields``, and a
    top-level page cut this way gets its paging fields restated.

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
        cut = _replace_at(payload, path, replacement)
        if record is not None:
            cut["truncated"] = True
            cut["truncated_fields"] = [*payload.get("truncated_fields", []), record]
            if len(path) == 1:
                _restate_paging(cut, payload, record["returned"], record["total"])
        cut_text = _encode(cut)
        if len(cut_text) >= len(text):
            break
        payload, text = cut, cut_text
    return payload, text
