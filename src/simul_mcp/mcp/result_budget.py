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
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# ~40 KB ≈ 10k tokens: large enough for a real listing, small enough that a
# single call cannot dominate a context window.
DEFAULT_RESULT_BUDGET_BYTES = 40_000

_DEFAULT_HINT = (
    "Result truncated to fit the response budget. Narrow the query — a more "
    "specific root_path, a smaller max_items/max_prims, or a filter — to see "
    "the rest."
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


def apply_result_budget(
    payload: Any,
    budget_bytes: int = DEFAULT_RESULT_BUDGET_BYTES,
    hint: str = _DEFAULT_HINT,
) -> Any:
    """Trim ``payload`` so its JSON encoding fits ``budget_bytes``.

    Only list fields are trimmed. A payload whose bulk is a single large string
    (a base64 capture, say) is left alone: guessing where to cut an opaque blob
    is more likely to corrupt it than to help, and those paths cap themselves at
    the point they encode.

    Returns the payload unchanged when it already fits.
    """
    if not isinstance(payload, dict):
        return payload
    if _encoded_size(payload) <= budget_bytes:
        return payload

    largest = _largest_list_field(payload)
    if largest is None:
        return payload

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
    return trimmed
