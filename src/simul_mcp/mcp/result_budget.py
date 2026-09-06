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

A payload whose bulk is a string cannot be trimmed the same way — guessing where
to cut an opaque blob is more likely to corrupt it than to help — so up to a
second, hard ceiling it is passed through and flagged. Above the hard ceiling
the string is replaced by a marker naming the field and its size: one call must
not be able to evict the conversation it was meant to inform.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# ~40 KB ≈ 10k tokens: large enough for a real listing, small enough that a
# single call cannot dominate a context window.
DEFAULT_RESULT_BUDGET_BYTES = 40_000

# ~400 KB ≈ 100k tokens. Nothing the caller can read is worth this much of its
# context; a string field this large is replaced rather than passed through.
HARD_RESULT_LIMIT_BYTES = 400_000

_DEFAULT_HINT = (
    "Result truncated to fit the response budget. Narrow the query — a more "
    "specific root_path, a smaller max_items/max_prims, or a filter — to see "
    "the rest."
)

_HARD_LIMIT_HINT = (
    "Field replaced: it exceeded the hard result limit. Write large output to "
    "a file and return its path, or return less of it."
)


def _encoded_size(payload: Any) -> int:
    return len(json.dumps(payload, default=str))


def _largest_list_field(payload: Dict[str, Any]) -> Optional[Tuple[str, List[Any]]]:
    """Return the (key, list) pair contributing the most encoded bytes."""
    candidates = [
        (key, value)
        for key, value in payload.items()
        if isinstance(value, list) and value
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: _encoded_size(item[1]))


def _largest_string_field(payload: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Return the (key, str) pair contributing the most encoded bytes."""
    candidates = [
        (key, value) for key, value in payload.items() if isinstance(value, str) and value
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: len(item[1]))


def apply_result_budget(
    payload: Any,
    budget_bytes: int = DEFAULT_RESULT_BUDGET_BYTES,
    hint: str = _DEFAULT_HINT,
    hard_limit_bytes: int = HARD_RESULT_LIMIT_BYTES,
) -> Any:
    """Trim ``payload`` so its JSON encoding fits ``budget_bytes``.

    Only list fields are trimmed to meet the budget. A payload whose bulk is a
    string is flagged with ``oversized_bytes`` and passed through — until it
    exceeds ``hard_limit_bytes``, at which point the largest string fields are
    replaced by ``{"truncated_field": name, "original_bytes": n, "hint": ...}``
    until the payload fits the hard limit.

    Args:
        payload: The tool's result; anything but a dict is returned unchanged.
        budget_bytes: Soft ceiling met by trimming lists.
        hint: Text placed in the truncation notice.
        hard_limit_bytes: Ceiling above which string fields are replaced.

    Returns:
        The payload unchanged when it already fits, else the trimmed copy.
    """
    if not isinstance(payload, dict):
        return payload
    if _encoded_size(payload) <= budget_bytes:
        return payload
    return _enforce_hard_limit(_trim_largest_list(payload, budget_bytes, hint), hard_limit_bytes)


def _trim_largest_list(payload: Dict[str, Any], budget_bytes: int, hint: str) -> Dict[str, Any]:
    """Cut the largest top-level list until the payload fits, or flag it oversized."""

    def _oversized(original: Dict[str, Any]) -> Dict[str, Any]:
        """Report an over-budget payload we could not usefully trim."""
        noted = dict(original)
        noted["oversized_bytes"] = _encoded_size(original)
        return noted

    largest = _largest_list_field(payload)
    if largest is None:
        return _oversized(payload)

    field, items = largest
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
        average = max(1, _encoded_size(items) // total)
        kept = max(0, min(total, remaining // average))
        while kept > 0 and _encoded_size(items[:kept]) > remaining:
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

    # Only keep the trim if it actually helped. The largest *top-level* list is
    # not always where the bulk is: get_isaac_prim_detail carries its weight in
    # dict values with one small "aspects" list, and prim info carries it in
    # "attributes" while the only list is "children". Trimming those drops
    # metadata the caller needs, reports a truncation that did not happen, and
    # leaves the payload over budget — larger, in fact, since the notice costs
    # more than the list did.
    if _encoded_size(trimmed) < _encoded_size(payload):
        return trimmed

    # Nothing worth trimming at the top level. Say the payload is oversized
    # rather than pretending it was cut; a caller acting on a false
    # "truncated" flag would go looking for data that is all still here.
    return _oversized(payload)


def _enforce_hard_limit(payload: Dict[str, Any], hard_limit_bytes: int) -> Dict[str, Any]:
    """Replace the largest string fields with markers until the payload fits."""
    while _encoded_size(payload) > hard_limit_bytes:
        largest = _largest_string_field(payload)
        if largest is None:
            return payload
        field, value = largest
        marker = {
            "truncated_field": field,
            "original_bytes": len(value.encode("utf-8")),
            "hint": _HARD_LIMIT_HINT,
        }
        if _encoded_size(marker) >= _encoded_size(value):
            # The remaining strings are all small; replacing them cannot help.
            return payload
        payload = dict(payload)
        payload[field] = marker
    return payload
