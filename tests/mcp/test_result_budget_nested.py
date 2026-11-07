"""The result budget reaches nested fields, keeps paging truthful and emits valid JSON.

Three ways a result used to slip past the budget:

- the hard limit only replaced top-level strings, so a prim's million-point
  attribute (nested under ``attributes``) crossed the wire whole, 51 MB;
- trimming a page left ``next_offset`` and ``count`` describing the page that
  was sent, so a client following ``next_offset`` skipped every item the trim
  dropped (630 of 1000 on one Isaac listing);
- ``json.dumps`` writes ``inf`` as the bare token ``Infinity``, which strict
  JSON parsers reject (``physics:breakForce`` defaults to it).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from simul.mcp.result_budget import (
    HARD_RESULT_LIMIT_BYTES,
    apply_result_budget,
    encode_result_budget,
)


def _prims(count: int) -> List[Dict[str, Any]]:
    return [
        {
            "path": f"/World/Environment/Building_{i}/Floor/Mesh_{i}",
            "name": f"Mesh_{i}",
            "type": "Mesh",
            "depth": 4,
            "child_count": 3,
        }
        for i in range(count)
    ]


def _encoded_size(payload: Any) -> int:
    return len(json.dumps(payload, default=str))


def _strict_loads(text: str) -> Any:
    def _reject(name: str) -> Any:
        raise ValueError(f"non-standard JSON constant {name}")

    return json.loads(text, parse_constant=_reject)


# ---------------------------------------------------------------------------
# The hard limit reaches nested fields
# ---------------------------------------------------------------------------


def test_nested_list_above_the_hard_limit_is_cut_where_it_sits() -> None:
    points = [[float(i), 0.5, -0.25] for i in range(200_000)]
    payload = {
        "success": True,
        "path": "/World/Terrain",
        "attributes": {
            "points": {"type": "point3f[]", "value": points},
            "doubleSided": {"type": "bool", "value": False},
        },
    }

    text = encode_result_budget(payload)
    result = _strict_loads(text)

    assert len(text) <= HARD_RESULT_LIMIT_BYTES
    kept = result["attributes"]["points"]["value"]
    assert 0 < len(kept) < len(points)
    assert kept == points[: len(kept)]
    assert result["attributes"]["doubleSided"] == {"type": "bool", "value": False}
    assert result["truncated"] is True
    assert result["truncated_fields"] == [
        {
            "field": "attributes.points.value",
            "returned": len(kept),
            "total": len(points),
            "hint": result["truncated_fields"][0]["hint"],
        }
    ]
    assert payload["attributes"]["points"]["value"] is points, "the caller's payload was mutated"
    assert len(payload["attributes"]["points"]["value"]) == 200_000


def test_nested_string_above_the_hard_limit_is_replaced_by_a_marker() -> None:
    payload = {"success": True, "result": {"stdout": "x" * 900_000, "code": 0}}

    result = apply_result_budget(payload)

    assert result["result"]["stdout"]["truncated_field"] == "result.stdout"
    assert result["result"]["stdout"]["original_bytes"] == 900_000
    assert result["result"]["code"] == 0
    assert _encoded_size(result) <= HARD_RESULT_LIMIT_BYTES


def test_mapping_of_many_small_entries_keeps_a_prefix() -> None:
    attributes = {f"attr_{i:05d}": {"type": "float", "value": i} for i in range(20_000)}
    payload = {"success": True, "attributes": attributes}

    result = apply_result_budget(payload)

    kept = result["attributes"]
    assert _encoded_size(result) <= HARD_RESULT_LIMIT_BYTES
    assert 0 < len(kept) < len(attributes)
    assert list(kept) == list(attributes)[: len(kept)]
    assert result["truncated_fields"][0]["field"] == "attributes"


def test_list_of_records_is_cut_by_record() -> None:
    payload = {"success": True, "data": {"results": _prims(20_000)}}

    result = apply_result_budget(payload)

    kept = result["data"]["results"]
    assert _encoded_size(result) <= HARD_RESULT_LIMIT_BYTES
    assert kept == payload["data"]["results"][: len(kept)]
    assert result["truncated_fields"][0]["field"] == "data.results"


def test_payload_under_the_hard_limit_keeps_its_nested_fields() -> None:
    """Between the budget and the hard limit a nested bulk is flagged, not cut."""
    payload = {"success": True, "info": {"values": list(range(30_000))}}

    result = apply_result_budget(payload)

    assert result["info"]["values"] == payload["info"]["values"]
    assert result["oversized_bytes"] == _encoded_size(payload)
    assert "truncated_fields" not in result


def test_encoding_matches_the_budgeted_payload_after_a_nested_cut() -> None:
    payload = {"success": True, "attributes": {"points": {"value": list(range(300_000))}}}

    assert encode_result_budget(payload) == json.dumps(apply_result_budget(payload), default=str)


# ---------------------------------------------------------------------------
# Valid JSON: non-finite floats
# ---------------------------------------------------------------------------


def test_non_finite_floats_become_strings() -> None:
    payload = {
        "success": True,
        "attributes": {"physics:breakForce": {"value": float("inf")}},
        "limits": [float("-inf"), 1.5, float("nan")],
    }

    text = encode_result_budget(payload)
    result = _strict_loads(text)

    assert result["attributes"]["physics:breakForce"]["value"] == "Infinity"
    assert result["limits"] == ["-Infinity", 1.5, "NaN"]
    assert text == json.dumps(apply_result_budget(payload), default=str)


def test_finite_payload_encoding_is_unchanged() -> None:
    payload = {"success": True, "values": [1.5, -2.0, 0.0], "name": "ok"}

    assert encode_result_budget(payload) == json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Paging stays truthful when the budget trims a page
# ---------------------------------------------------------------------------


def _page(entries: List[Dict[str, Any]], offset: int, limit: int) -> Dict[str, Any]:
    """An Isaac-style paged listing (count, total, offset, next_offset)."""
    page = entries[offset:offset + limit]
    truncated = offset + len(page) < len(entries)
    return {
        "count": len(page),
        "total": len(entries),
        "offset": offset,
        "applied_limit": limit,
        "truncated": truncated,
        "next_offset": offset + len(page) if truncated else None,
        "prims": page,
    }


def test_trimmed_page_points_next_offset_past_the_items_kept() -> None:
    """A 1000-item page cut to fit used to keep next_offset 1000 (Isaac audit F13)."""
    page = apply_result_budget(_page(_prims(3000), offset=1000, limit=1000))

    kept = len(page["prims"])
    assert 0 < kept < 1000
    assert page["next_offset"] == 1000 + kept
    assert page["count"] == kept
    assert page["total"] == 3000
    assert page["truncated"] is True


def test_following_next_offset_through_trimmed_pages_reaches_every_item() -> None:
    entries = _prims(2500)
    seen: List[str] = []
    offset: Any = 0
    while offset is not None:
        page = apply_result_budget(_page(entries, offset=offset, limit=1000))
        seen.extend(prim["path"] for prim in page["prims"])
        offset = page["next_offset"]

    assert seen == [prim["path"] for prim in entries]


def test_last_page_cut_by_the_budget_gains_a_next_offset() -> None:
    """The final page said next_offset None; once cut, the rest must be reachable."""
    page = apply_result_budget(_page(_prims(1000), offset=0, limit=1000))

    assert page["next_offset"] == len(page["prims"]) < 1000


def test_grand_total_count_is_not_rewritten() -> None:
    """``count`` that is not the page's length (a match total) stays as sent."""
    payload = {"count": 9000, "offset": 0, "next_offset": 5000, "results": _prims(5000)}

    page = apply_result_budget(payload)

    assert page["count"] == 9000
    assert page["next_offset"] == len(page["results"])


def test_parallel_page_lists_resume_after_the_cut() -> None:
    """list_isaac_physics_objects pages three lists with one offset."""
    payload = {
        "offset": 0,
        "applied_limit": 1000,
        "truncated": True,
        "next_offset": 1000,
        "rigid_bodies": _prims(20),
        "colliders": _prims(1000),
        "joints": [],
    }

    page = apply_result_budget(payload)

    assert page["truncation"]["field"] == "colliders"
    assert page["next_offset"] == len(page["colliders"]) < 1000


def test_unpaged_payload_gains_no_paging_fields() -> None:
    page = apply_result_budget({"success": True, "prims": _prims(5000)})

    assert "next_offset" not in page
    assert "count" not in page
